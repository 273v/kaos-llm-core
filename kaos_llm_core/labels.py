"""Label and ``LabelSet`` types for classification programs.

A :class:`LabelSet` is the declarative source of truth for the label
space of any classification program in this package. The same object
flows through zero-shot, few-shot, prototype, retrieval, hierarchical,
multi-label, and chunked classifiers without translation.

Design notes
------------
- :class:`Label` and :class:`LabelSet` are Pydantic models because they
  flow across boundaries: CLI, MCP tools, JSON config, agent
  invocations.
- ``LabelSet`` carries policy flags (``exclusive``, ``allow_abstain``,
  ``hierarchical``) rather than overloading subclasses. Different
  classifier shapes consume the same data with different semantics.
- A hierarchical label set is represented flatly: each :class:`Label`
  carries an optional ``parent`` field. The tree is reconstructed by
  consumers via :meth:`LabelSet.children`. This keeps
  ``hierarchical=True`` and ``hierarchical=False`` identical on disk
  and lets a flat label set be promoted to a tree later by setting
  the flag and populating ``parent``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from kaos_core.types.content import KaosModel
from pydantic import Field, model_validator

ABSTAIN_LABEL = "__abstain__"
"""Canonical name returned by classifiers that abstain.

Reserved; user-supplied labels named ``__abstain__`` are rejected at
:class:`LabelSet` construction so callers can distinguish "no label
fits" from "the label literally named ``__abstain__``".
"""


class Label(KaosModel):
    """A single classification target.

    Attributes:
        name: Stable identifier. Used as the on-the-wire label string
            and as the key in score maps. Must be non-empty.
        description: Optional natural-language description. Surfaced
            to LLM classifiers in the prompt and to embedding
            classifiers as the prototype text. Defaults to ``name``
            when not provided.
        examples: Optional verbatim example inputs that demonstrate
            the label. Surfaced in few-shot prompts and prototype
            embedding pools.
        parent: Optional name of a parent label, for hierarchical
            taxonomies. ``None`` means the label is at the root.
    """

    name: str = Field(min_length=1)
    description: str | None = None
    examples: list[str] = Field(default_factory=list)
    parent: str | None = None

    @property
    def prompt_text(self) -> str:
        """Return the text a classifier should use for this label.

        Falls back to ``name`` when no description is set; LLM and
        embedding classifiers should call this rather than reading
        ``description`` directly so both shapes get the same input.
        """
        return self.description if self.description else self.name


class LabelSet(KaosModel):
    """Declarative description of a label space.

    Attributes:
        labels: Ordered sequence of :class:`Label`. Order is stable
            and reflected in any returned score map.
        exclusive: ``True`` (default) for single-label classification;
            ``False`` for multi-label classification.
        allow_abstain: When ``True``, classifiers are permitted to
            return zero labels (or the canonical ``ABSTAIN_LABEL``).
            When ``False``, the classifier must assign at least one
            label from :attr:`labels`.
        hierarchical: When ``True``, :attr:`labels` describes a tree
            via :attr:`Label.parent`. ``False`` ignores ``parent``.

    Construction validation:
        - Label names must be unique.
        - No label may be named ``__abstain__`` (reserved).
        - When ``hierarchical=True``, every ``parent`` reference must
          resolve to a label in the same set, and the parent graph
          must be acyclic.
    """

    labels: list[Label] = Field(min_length=1)
    exclusive: bool = True
    allow_abstain: bool = True
    hierarchical: bool = False

    @model_validator(mode="after")
    def _validate_label_set(self) -> LabelSet:
        names = [label.name for label in self.labels]
        if len(names) != len(set(names)):
            duplicate = next(n for n in names if names.count(n) > 1)
            raise ValueError(f"duplicate label name: {duplicate!r}")
        if ABSTAIN_LABEL in names:
            raise ValueError(f"label name {ABSTAIN_LABEL!r} is reserved for the abstain marker")
        if self.hierarchical:
            known = set(names)
            for label in self.labels:
                if label.parent is not None and label.parent not in known:
                    raise ValueError(f"label {label.name!r} has unknown parent {label.parent!r}")
            self._check_acyclic()
        return self

    def _check_acyclic(self) -> None:
        parent_of: dict[str, str | None] = {label.name: label.parent for label in self.labels}
        for start_name in parent_of:
            visited: set[str] = set()
            current: str | None = start_name
            while current is not None:
                if current in visited:
                    raise ValueError(f"cycle in hierarchical label set involving {current!r}")
                visited.add(current)
                current = parent_of.get(current)

    # ----- container protocol ---------------------------------------
    #
    # ``__iter__`` intentionally overrides :meth:`BaseModel.__iter__`
    # (which yields ``(field_name, value)`` tuples for v1 compatibility).
    # The label-set is conceptually a sequence of :class:`Label`, so
    # iteration over a :class:`LabelSet` yields labels — matching the
    # mental model callers already have. Field-tuple iteration is still
    # available via ``model_dump().items()`` when truly needed.

    def __iter__(self) -> Iterator[Label]:  # ty: ignore[invalid-method-override]
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return any(label.name == name for label in self.labels)

    # ----- convenience accessors ------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Return the ordered tuple of label names."""
        return tuple(label.name for label in self.labels)

    def by_name(self, name: str) -> Label:
        """Return the :class:`Label` with the given name.

        Raises:
            KeyError: If ``name`` is not present.
        """
        for label in self.labels:
            if label.name == name:
                return label
        raise KeyError(name)

    def children(self, parent_name: str | None) -> tuple[Label, ...]:
        """Return labels whose ``parent`` equals ``parent_name``.

        ``parent_name=None`` returns the root labels. The method works
        regardless of :attr:`hierarchical`; the flag only affects
        whether consumers should call it.
        """
        return tuple(label for label in self.labels if label.parent == parent_name)

    def roots(self) -> tuple[Label, ...]:
        """Return the root labels (those without a parent)."""
        return self.children(None)

    def validate_picks(self, picks: Sequence[str]) -> list[str]:
        """Return ``picks`` filtered to known labels, in input order.

        Unknown labels are silently dropped. Use :meth:`assert_picks`
        when the caller wants a hard failure instead.
        """
        known = set(self.names)
        return [p for p in picks if p in known]

    def assert_picks(self, picks: Sequence[str]) -> list[str]:
        """Return ``list(picks)`` after asserting every name is known.

        Raises:
            KeyError: If any pick is not in the label set.
        """
        known = set(self.names)
        unknown = [p for p in picks if p not in known]
        if unknown:
            raise KeyError(f"unknown labels: {unknown!r}")
        return list(picks)

    # ----- constructors --------------------------------------------

    @classmethod
    def from_names(
        cls,
        names: Sequence[str],
        *,
        exclusive: bool = True,
        allow_abstain: bool = True,
    ) -> LabelSet:
        """Build a flat (non-hierarchical) :class:`LabelSet` from names only.

        Convenience constructor for the very common case where the
        caller has nothing but a list of label strings (e.g., from a
        CLI flag).
        """
        return cls(
            labels=[Label(name=n) for n in names],
            exclusive=exclusive,
            allow_abstain=allow_abstain,
            hierarchical=False,
        )


__all__ = [
    "ABSTAIN_LABEL",
    "Label",
    "LabelSet",
]
