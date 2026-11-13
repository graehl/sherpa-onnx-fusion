---
slug: nested-option-collections
noticed: 2026-11-12
where: sherpa-onnx/csrc/parse-options.h, sherpa-onnx/csrc/parse-options.cc
---

# Typed collections of nested configuration objects

## Desired outcome

User-directed, 2026-11-12: define a typed options struct once, hold a
collection such as `std::vector<ScorerOptions>`, and configure its nested
fields through JSON or indexed CLI names without hand-writing separate
field mappings for every transport. The facility should work for other
option collections too. This is intended functionality, not a claim that
the existing parser violates its contract.

The immediate consumer is [shared beam search and scorer fusion](../../topics/beam-search-and-scorer-fusion.md),
which needs multiple independently configured scorers. Another possible
consumer is an ordered list of FST inputs: callers could supply separate
graphs and request composition rather than build one combined file outside
the application.

## Mechanism before this addition

`ParseOptions::Register` accepts scalar field pointers. Its prefix wrapper
forms dotted names, and `SetOption` assigns an exact registered key.
It had no generic registration for vectors of nested structs; repeating
one scalar key overwrites that field. Indexed names can be registered
explicitly, but collection creation and storage lifetime are then the
caller's responsibility. Existing `--config` files contain CLI-style
assignments, not JSON.

## Possibilities to assess

The selected direction is recorded in
[option collections](../../topics/json-option-collections.md): indexed
command-line paths through a small parser hook, with JSON applied as the
same paths. The alternatives below are the background it was chosen from.

- Wrap existing registration in a reusable typed description that also
  handles nested objects and collections, with JSON and/or CLI adapters.
- Extend the current parser with collection registration. For indexed
  options such as `--fusion.scorers.0.weight=0.3`, discover and allocate
  entries before binding their fields, or use accessors that do not retain
  pointers invalidated by vector growth.
- Use JSON for collections while preserving ordinary scalar CLI options.
  Keep one typed configuration model and shared defaults and validation;
  avoid a scorer-specific JSON parser duplicating every field declaration.

Graehl's usual reference is the `configure.hpp` pattern from his other C++
work: one visitor-style description of a struct's fields serving several
readers. Consider it when evaluating implementation approaches, but prefer a small
adaptation close to Sherpa's current conventions. No decision has been
made to fork Kaldi, contribute this upstream, or replace the configuration
framework. A local wrapper or extension is an option, not a commitment.

## Status

The mechanism is implemented and tested with a non-scorer fixture, and the
recognizer's scorer configuration uses it; see the topic. The multiple-FST
use below remains a possible second consumer.

## Decisions still open

The [topic's rules](../../topics/json-option-collections.md#rules) settle
collection ordering, index gaps, duplicate assignments, unknown-field
errors, and precedence between files and CLI overrides.

For multiple FST inputs, configuration is only the entry point. The consuming
code must define composition order, label/symbol compatibility, and eager
versus lazy composition and its ownership. A list of FST paths alone does
not settle those semantics. Keep composition outside the generic parser.

## Completion evidence

Demonstrate the chosen mechanism with a nested struct collection and a
second non-scorer use case or fixture. Check that supplied values reach the
typed objects, defaults and overrides agree, malformed fields fail clearly,
and collection growth does not invalidate registered storage. Document the
chosen syntax and compatibility behavior before adopting it for fusion.

Contributing-model: gpt-6-astra
