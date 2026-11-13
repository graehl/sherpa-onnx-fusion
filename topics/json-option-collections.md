# Option collections: indexed paths and JSON

> Configure a vector of option structs through indexed command-line paths
> such as `--scorer.1.weight`, and through JSON that maps onto the same
> paths, using each struct's ordinary `Register(ParseOptions*)`.

Topic: `json-option-collections`

## Scope and status

This is the convention for configuring the multiple
[fusion scorers](beam-search-and-scorer-fusion.md). The mechanism is
implemented and tested with a fixture collection
(`sherpa-onnx/csrc/option-collection.h`, `option-json.h`,
`option-collection-test.cc`), and the offline recognizer's `--scorer.*` and
`--fusion-config` options use it. It works for any `std::vector` of a struct that has
the project's usual `Register(ParseOptions *)`; a list of FST inputs is
another possible consumer. Existing options and `--config` files keep their
behavior. The configs it serves are simple: scalars, nested structs, and
vectors of structs. Maps, config-file inclusion, and path expansion are out
of scope.

## What a user writes

One scorer, with the index left out:

```text
--scorer.type=kenlm --scorer.model=lm.bin --scorer.weight=0.6
```

Several, by zero-based index:

```text
--scorer.0.type=kenlm    --scorer.0.model=lm.bin       --scorer.0.weight=0.6
--scorer.1.type=onnx-rnn --scorer.1.model=lstm.onnx    --scorer.1.weight=0.3
--scorer.1.extra='{"vocab":"units.txt"}'
```

The same in a JSON file named by an option:

```json
{ "scorer": [
    { "type": "kenlm", "model": "lm.bin", "weight": 0.6 },
    { "type": "onnx-rnn", "model": "lstm.onnx", "weight": 0.3,
      "extra": { "vocab": "units.txt" } } ] }
```

`--scorer.model` means `--scorer.0.model`. JSON keys are the option names,
and a JSON array position is the index. Where a collection is expected,
JSON may give a single object in place of an array, meaning a list of one:
`{ "scorer": { "type": "kenlm", "model": "lm.bin" } }`. A `--config` file can hold the
indexed lines too, since it holds ordinary options.

## Mechanism

**The struct is unchanged in kind.** An element type registers its fields on
a `ParseOptions`, validates, and prints, like every other config here.
Nested structs register through the existing prefix constructor. Defaults
live in the struct. There is one list of fields and no JSON schema.

**The parser has one hook.** `ParseOptions` treats an unknown key as an
error. `ParseOptions::RegisterHandler` adds a handler for a name: when a
key is unknown and is that name or begins with it and a dot, the parser
passes the rest of the key and the value to the handler before failing.
With `ParseOptions::Assign`, which sets one option by name, this is the
only change to an upstream file, and it is additive.

**A collection handler assigns one path at a time.**
`RegisterOptionCollection` installs it. For `scorer.1.weight=0.3` it grows the vector to two default-constructed
elements, registers element 1 on a local `ParseOptions`, and gives that
parser `--weight=0.3`. The existing string conversions and nested prefixes
apply unchanged, and no pointer into the vector outlives the call, so
growth cannot invalidate anything. A key that the element does not register
is an error naming the full path.

**JSON is applied as option paths.** The file is ordinary nested JSON.
`ApplyJsonOptions` walks it: an object member appends `.name` to the path, an array
element appends `.index`, and each scalar becomes one assignment given to
the same parser. JSON, the command line, and `--config` therefore share
names, defaults, validation, and errors. A JSON key containing `.` is
rejected, so each option has one JSON spelling. An object assigned to a string option is
passed through as JSON text; that is how a scorer's `extra` reaches its
factory. Use the project's existing `nlohmann::json` for parsing and keep
it out of scorer factory interfaces.

## Rules

- Order: options apply in command-line order, and a later assignment to
  the same path wins. A JSON file named by an option
  (`RegisterJsonOptionsFile`) is applied at that option's position, so put
  it before the options that override it. `Validate()` runs after all of
  them.
- Indices are zero-based decimal, below 256, and need not arrive in order.
  Assigning index N creates elements 0 through N with default values; an element
  nobody assigned then fails its own validation (a scorer with no type).
- A JSON array replaces nothing by itself: it assigns the paths it
  contains. Giving both a JSON file and indexed options for one element is
  allowed and follows the order above.
- Reject duplicate keys within one JSON object while parsing, before a
  later value silently replaces an earlier one. `null` is an error.
- JSON values pass through the command line's string conversions, so
  `"weight": "0.3"` is accepted as well as `0.3`, and `true` or `1` for a
  bool. Non-finite numbers are rejected.
- An invalid path or value is reported with the full path and, for JSON,
  the filename, and ends the program as an invalid option does today.
  Library callers that build the structs directly are unaffected.
- Usage output lists an element's options once, as `--scorer.N.<name>`.

## Checks

Exercise a fixture collection that is not the scorer list: one and several
elements, the elided index, out-of-order indices, nested struct fields,
defaults, a later assignment overriding an earlier one, a single JSON
object in place of an array, JSON followed by
command-line overrides, an object passed through as text, an unknown field,
a duplicate JSON key, and a malformed value. Then check that a scorer
configuration reaches factory selection, common options, per-scorer
weights, and the factory's `extra` text through the real recognizer
construction path.

## Design decisions

- **JSON applied as option paths** (vs. a typed JSON reader visiting
  templated `Register` bodies): one field list and one set of conversions
  and errors for every input form, with about a hundred lines of new code
  and no change to existing config structs. Accepts looser typing of JSON
  values and exit-on-error instead of leaving a previous config intact.
- **Indexed paths through a parser hook** (vs. JSON only, or a dedicated
  set of single-scorer flags): every vector of options gets a command-line
  form from its element's existing registration, and the single-element
  case needs no second spelling of the fields.
- **Index may be omitted for element 0** (vs. always required): the common
  one-scorer command stays short. Accepts two spellings of element 0.
- **Assign through a local parser per path** (vs. registering element
  fields on the main parser): no pointers are retained, so the vector can
  grow, and the main parser needs to know nothing about collections.

[The collection sketch](../gaps/sketches/nested-option-collections.md)
tracks the remaining work.
