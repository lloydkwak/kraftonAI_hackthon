"""The graph validator: the same checks intake runs, runnable locally.

    python -m wam.validator <submission_dir> [--static-only] [--json]

For a submission directory it checks, and reports the numbers behind:

* the manifest -- ranges, the `N <= 4K` coupling, the `ctx` byte cap;
* the graph set (all seven graphs), each graph's inputs a subset of the contract's and its
  outputs exact (`contract.io_spec`);
* operators: default ONNX domain only, no custom domains, no `Random*` (randomness is the
  runner's) -- in the main graph and in every `Loop` / `Scan` / `If` body;
* parameters: every stored tensor -- `initializer`s, `sparse_initializer`s and the tensor of
  every `Constant` node, in the main graph and in every subgraph -- summed across graphs,
  **shared tensors counted once** by content hash, against the 0.5B limit;
* FLOPs per graph from static shapes at `B = 1`, reported, not capped: `predict_step` per call,
  `visual_decode` per frame, `visual_encode + observe + act` per control step, and the per-frame
  prediction cost `N x predict_step / K`.  Control flow is counted: a `Loop` body times its
  constant trip count, a `Scan` body times its sequence length, the costlier `If` branch.  A
  graph whose shapes cannot be made static (a data-dependent shape, a loop without a constant
  trip count) is **rejected** rather than under-counted;
* determinism: a recorded call re-run gives the same output within 1e-4;
* statelessness across the batch: row `i` of a batch equals the single-row call -- a graph
  that mixes rows, or carries state, fails this;
* the execution contract on a random probe: shapes, finite values, RGB range, actions in [-1, 1];
* `encode` wall-clock against the 1 s limit (an error under the pinned CUDA runtime, a warning
  on any other provider).

The FLOP counter counts multiply-adds as 2 FLOPs for Conv / ConvTranspose / MatMul / Gemm and
one FLOP per output element for everything else, from inferred static shapes.  It is checked
against `stem.Stem.cost` in testing to within a few percent; the point is that it is one
published counter, not that it agrees with any vendor's.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import contract as CT

FORBIDDEN_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
                 "Multinomial", "Bernoulli"}
ALLOWED_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


class ValidationError(ValueError):
    """A graph the validator cannot count: tensor data it cannot read (external data
    missing), a shape that is not static, a loop whose trip count is not a constant.
    `validate` turns it into a rejection carrying the message; you fix the graph."""


@dataclass
class Report:
    ok: bool = True
    errors: list = field(default_factory=list)      # hard failures: the submission is rejected
    warnings: list = field(default_factory=list)    # soft findings
    numbers: dict = field(default_factory=dict)

    def fail(self, msg):
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def to_json(self):
        return dict(ok=self.ok, errors=self.errors, warnings=self.warnings, numbers=self.numbers)


# --------------------------------------------------------------------------------
# Static: operators, parameters, FLOPs
# --------------------------------------------------------------------------------


def _load(path):
    """`onnx.load`, with a graph the library cannot read turned into a `ValidationError`:
    an external data file (the `.bin` beside the `.onnx`) that was not shipped, or a
    file that is not ONNX, is a rejected submission, not a traceback."""
    import onnx
    from google.protobuf.message import DecodeError
    try:
        return onnx.load(path)
    except onnx.checker.ValidationError as e:
        raise ValidationError(f"{os.path.basename(path)}: cannot read the graph's tensor data ({e}) -- ship every "
                              f"external data file beside the graph, or embed the weights in the .onnx") from e
    except (DecodeError, OSError) as e:
        raise ValidationError(f"{os.path.basename(path)}: not a readable ONNX file ({type(e).__name__}: {e})") from e


def _attr(node, name, default=None):
    from onnx import helper
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _subgraphs(node):
    """The graphs an `If` / `Loop` / `Scan` node carries as attributes."""
    from onnx import AttributeProto
    for a in node.attribute:
        if a.type == AttributeProto.GRAPH:
            yield a.g
        elif a.type == AttributeProto.GRAPHS:
            yield from a.graphs


def _all_nodes(graph):
    """Every node of `graph` and, recursively, of its subgraphs."""
    for node in graph.node:
        yield node
        for sub in _subgraphs(node):
            yield from _all_nodes(sub)


def check_operators(model, name: str, rep: Report):
    ops = {}
    for node in _all_nodes(model.graph):
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
        if node.domain not in ALLOWED_DOMAINS:
            rep.fail(f"{name}: operator {node.op_type} in domain {node.domain!r} (standard ONNX operators only)")
        if node.op_type in FORBIDDEN_OPS:
            rep.fail(f"{name}: {node.op_type} draws randomness inside the graph (the runner supplies randomness)")
    for imp in model.opset_import:
        if imp.domain not in ALLOWED_DOMAINS:
            rep.fail(f"{name}: opset import from domain {imp.domain!r} (standard ONNX domains only)")
    return ops


# -- parameters -------------------------------------------------------------------


def _attr_array(a):
    """The tensor a `Constant` node's attribute carries, as numpy: `value`, `value_float(s)`,
    `value_int(s)`, `value_string(s)`.  `None` for anything else (`sparse_value` is read
    separately)."""
    from onnx import AttributeProto, numpy_helper
    if a.type == AttributeProto.TENSOR:
        return numpy_helper.to_array(a.t)
    if a.type == AttributeProto.FLOAT:
        return np.array(a.f, np.float32)
    if a.type == AttributeProto.FLOATS:
        return np.array(list(a.floats), np.float32)
    if a.type == AttributeProto.INT:
        return np.array(a.i, np.int64)
    if a.type == AttributeProto.INTS:
        return np.array(list(a.ints), np.int64)
    if a.type == AttributeProto.STRING:
        return np.array(a.s, dtype=object)
    if a.type == AttributeProto.STRINGS:
        return np.array(list(a.strings), dtype=object)
    return None


def _content_key(arr) -> str:
    """The dedup key of a tensor: content bytes + shape + dtype (identical tensors in
    several graphs are one set of weights)."""
    if arr.dtype.kind in ("O", "U", "S"):
        raw = b"\0".join(s if isinstance(s, bytes) else str(s).encode() for s in arr.ravel().tolist())
    else:
        raw = arr.tobytes()
    return hashlib.sha1(raw).hexdigest() + str(arr.shape) + str(arr.dtype)


def _sparse_entry(sp, read):
    """(name, dense element count, key) of a SparseTensorProto: it stores `values` at
    `indices` inside `dims`, and the cap counts the dense extent it stands for."""
    from onnx import numpy_helper
    vals = read(lambda: numpy_helper.to_array(sp.values), sp.values.name)
    idx = read(lambda: numpy_helper.to_array(sp.indices), sp.values.name)
    size = int(np.prod(sp.dims)) if len(sp.dims) else 1
    key = hashlib.sha1(vals.tobytes() + idx.tobytes()).hexdigest() + str(list(sp.dims)) + str(vals.dtype) + "sparse"
    return sp.values.name, size, key


def _stored_tensors(graph, where: str):
    """Every tensor stored in `graph` and, recursively, its subgraphs, as
    `(name, element count, dedup key)`: `initializer`s, `sparse_initializer`s (dense
    extent) and the tensor attribute of every `Constant` node -- weights count wherever a
    graph keeps them, whatever their dtype."""
    from onnx import AttributeProto, numpy_helper

    def read(fn, what):
        try:
            return fn()
        except Exception as e:                                       # noqa: BLE001
            raise ValidationError(f"{where}: cannot read the data of tensor {what!r} ({type(e).__name__}: {e}) "
                                  f"-- external data missing?") from e

    for init in graph.initializer:
        arr = read(lambda: numpy_helper.to_array(init), init.name)
        yield init.name, int(arr.size), _content_key(arr)
    for sp in graph.sparse_initializer:
        yield _sparse_entry(sp, read)
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            for a in node.attribute:
                if a.type == AttributeProto.SPARSE_TENSOR:
                    yield _sparse_entry(a.sparse_tensor, read)
                    continue
                arr = read(lambda: _attr_array(a), node.output[0])
                if arr is not None:
                    yield node.output[0], int(arr.size), _content_key(arr)
        for sub in _subgraphs(node):
            yield from _stored_tensors(sub, where)


def parameter_count(models: dict) -> tuple:
    """Total stored scalars with shared tensors counted once (by content hash), and the
    per-graph totals.  A stored tensor is every `initializer` and `sparse_initializer` of
    the main graph and of every `Loop` / `Scan` / `If` body, and the tensor of every
    `Constant` node in any of them; elements are counted whatever the dtype.  Raises
    `ValidationError` when a tensor's data cannot be read (external data not shipped)."""
    seen = set()
    total = 0
    per = {}
    for name, m in models.items():
        n = 0
        for _tname, size, key in _stored_tensors(m.graph, name):
            n += size
            if key not in seen:
                seen.add(key)
                total += size
        per[name] = n
    return total, per


# -- static shapes ------------------------------------------------------------------

# Shape arithmetic an exporter emits around `Reshape` / `Expand` / `Slice`; folded when
# every input is a constant (and small) so that the shapes downstream become static.
_FOLDABLE = {"Gather", "GatherElements", "Unsqueeze", "Squeeze", "Concat", "Slice", "Cast", "Equal", "Where",
             "ConstantOfShape", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Max", "Min", "Range", "Reshape",
             "Expand", "Identity", "Not", "And", "Or", "Less", "Greater", "LessOrEqual", "GreaterOrEqual",
             "Floor", "Ceil", "Mod", "ReduceProd", "ReduceSum", "ReduceMax", "ReduceMin", "Transpose", "Flatten",
             "Split", "Clip"}
_FOLD_MAX_ELEMENTS = 4096
_FOLD_ROUNDS = 8
_INT_TYPES = {2, 3, 4, 5, 6, 7, 9, 12, 13}                          # TensorProto UINT8, INT8, UINT16, INT16, INT32, INT64, BOOL, UINT32, UINT64
_ZERO_COST = {"Constant", "Shape", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity", "Cast",
              "Concat", "Slice", "Gather", "Split", "Expand", "ConstantOfShape"}
_CONTROL_FLOW = {"Loop", "Scan", "If"}


def _static_dims(vi):
    """The dims of a value_info as ints; `None` when it has no tensor shape or a symbolic dim."""
    if not vi.type.HasField("tensor_type") or not vi.type.tensor_type.HasField("shape"):
        return None
    out = []
    for d in vi.type.tensor_type.shape.dim:
        if not d.HasField("dim_value"):
            return None
        out.append(int(d.dim_value))
    return out


def _bind_batch(graph, batch: int):
    """Every dynamic dim of the graph's inputs and outputs (the contract's `B`) becomes `batch`."""
    for vi in list(graph.input) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                if not d.HasField("dim_value"):
                    d.dim_value = batch


def _set_dims(vi, dims):
    tt = vi.type.tensor_type
    tt.ClearField("shape")
    tt.shape.SetInParent()
    for d in dims:
        tt.shape.dim.add().dim_value = int(d)


def _scope(graph, outer_shapes: dict, outer_known: dict) -> tuple:
    """What is visible inside `graph`: the enclosing scopes' static shapes and constant values,
    plus its own inputs / outputs / value_info / initializers / `Constant` nodes.  `shapes`
    maps a name to its static dims or `None`; `known` holds the small constants (shape
    arithmetic, trip counts), never the weights."""
    from onnx import AttributeProto, TensorProto, numpy_helper
    shapes, known = dict(outer_shapes), dict(outer_known)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        s = _static_dims(vi)
        if s is not None:
            shapes[vi.name] = s
        else:
            shapes.setdefault(vi.name, None)
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)
        if int(np.prod(init.dims)) <= _FOLD_MAX_ELEMENTS and init.data_location != TensorProto.EXTERNAL:
            known[init.name] = numpy_helper.to_array(init)
    for sp in graph.sparse_initializer:
        shapes[sp.values.name] = list(sp.dims)
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            for a in node.attribute:
                if a.type == AttributeProto.TENSOR:                  # dims from the proto: a big one may have been stripped
                    shapes[node.output[0]] = list(a.t.dims)
                    if int(np.prod(a.t.dims)) <= _FOLD_MAX_ELEMENTS:
                        known[node.output[0]] = numpy_helper.to_array(a.t)
                elif a.type == AttributeProto.SPARSE_TENSOR:
                    shapes[node.output[0]] = list(a.sparse_tensor.dims)
                else:
                    arr = _attr_array(a)
                    if arr is not None and arr.dtype.kind != "O":
                        shapes[node.output[0]] = list(arr.shape)
                        known[node.output[0]] = arr
    return shapes, known


def _bind_body_inputs(node, shapes: dict) -> bool:
    """Give a `Loop` / `Scan` body's inputs the static shapes of the outer tensors they
    carry when the body declares none (or symbolic ones): `Loop` -- iteration counter and
    condition are scalars, the carried values take the initial values' shapes; `Scan` --
    state values take the initial states' shapes, scan inputs lose their scan axis."""
    from onnx import TensorProto
    body = _attr(node, "body")
    if body is None:
        return False
    if node.op_type == "Loop":
        specs = [([], TensorProto.INT64), ([], TensorProto.BOOL)] + [(shapes.get(n), 0) for n in node.input[2:]]
    elif node.op_type == "Scan":
        n_scan = int(_attr(node, "num_scan_inputs", 0) or 0)
        n_state = len(node.input) - n_scan
        axes = list(_attr(node, "scan_input_axes", []) or [0] * n_scan)
        specs = [(shapes.get(n), 0) for n in node.input[:n_state]]
        for n, ax in zip(node.input[n_state:], axes):
            s = shapes.get(n)
            specs.append((None if not s else [d for k, d in enumerate(s) if k != ax % len(s)], 0))
    else:
        return False
    changed = False
    for vi, (s, elem) in zip(body.input, specs):
        tt = vi.type.tensor_type
        if elem and tt.elem_type == TensorProto.UNDEFINED:
            tt.elem_type = elem
            changed = True
        if s is None or _static_dims(vi) is not None:
            continue
        if tt.HasField("shape") and len(tt.shape.dim) != len(s):
            continue
        _set_dims(vi, s)
        changed = True
    return changed


def _evaluate(node, known: dict, opsets: dict):
    """Run one shape-arithmetic node on its constant inputs with ONNX's reference evaluator.
    `None` when it cannot be run or would produce more than a shape's worth of data."""
    from onnx.reference import ReferenceEvaluator
    feeds = {i: known[i] for i in node.input if i}
    if node.op_type == "ConstantOfShape" and int(np.prod(feeds[node.input[0]])) > _FOLD_MAX_ELEMENTS:
        return None
    if node.op_type == "Expand" and int(np.prod(feeds[node.input[1]])) > _FOLD_MAX_ELEMENTS:
        return None
    if node.op_type == "Range":
        start, limit, delta = (float(np.asarray(feeds[i]).reshape(-1)[0]) for i in node.input[:3])
        if delta == 0 or (limit - start) / delta > _FOLD_MAX_ELEMENTS:
            return None
    try:
        out = ReferenceEvaluator(node, opsets=opsets).run(None, feeds)
    except Exception:                                                # noqa: BLE001
        return None
    return [np.asarray(o) for o in out]


def _fold_graph(graph, outer_shapes: dict, outer_known: dict, opsets: dict) -> bool:
    """In `graph` and its subgraphs, replace every shape computation whose result is fixed
    at this batch size by a constant: `Shape` / `Size` of a tensor with a static shape, and
    any `_FOLDABLE` node whose inputs are all constants.  The result becomes an initializer,
    which the next shape-inference pass reads (`Reshape`, `Expand`, `Slice`...).  Returns
    whether anything changed."""
    from onnx import numpy_helper
    shapes, known = _scope(graph, outer_shapes, outer_known)
    changed = False
    remove, new_inits = [], []
    for idx, node in enumerate(graph.node):
        if node.op_type in _CONTROL_FLOW:
            changed |= _bind_body_inputs(node, shapes)
            for sub in _subgraphs(node):
                changed |= _fold_graph(sub, shapes, known, opsets)
            continue
        if node.op_type == "Constant" or not node.output:
            continue
        vals = None
        if node.op_type in ("Shape", "Size") and node.input:
            src = node.input[0]
            dims = list(known[src].shape) if src in known else shapes.get(src)
            if dims is not None:
                if node.op_type == "Shape":
                    start, end = int(_attr(node, "start", 0)), _attr(node, "end", None)
                    vals = [np.array(dims[start:] if end is None else dims[start:int(end)], np.int64)]
                else:
                    vals = [np.array(int(np.prod(dims)) if dims else 1, np.int64)]
        elif node.op_type in _FOLDABLE and node.input and all(i == "" or i in known for i in node.input):
            # shape arithmetic is integer / boolean; float constant arithmetic is real work and stays counted
            ints = all(known[i].dtype.kind in "iub" for i in node.input if i)
            if ints or (node.op_type == "Cast" and _attr(node, "to") in _INT_TYPES):
                vals = _evaluate(node, known, opsets)
        if vals is None or len(vals) != len(node.output) or any(v.size > _FOLD_MAX_ELEMENTS for v in vals):
            continue
        for o, v in zip(node.output, vals):
            if o:
                s = shapes.get(o)
                if s is not None and int(np.prod(s)) == v.size:
                    v = v.reshape(s)                                 # the inferred shape is normative
                known[o] = v
                shapes[o] = list(v.shape)
                new_inits.append(numpy_helper.from_array(v, o))          # not ascontiguousarray: it turns a 0-d into a 1-d
        remove.append(idx)
        changed = True
    for idx in reversed(remove):
        del graph.node[idx]
    graph.initializer.extend(new_inits)
    return changed


def _strip_weights(graph):
    """Drop the data of every tensor larger than a shape's worth, in `graph` and its subgraphs.
    The copy used for shape inference needs dims and dtypes, plus the values of the small
    shape constants; inference serialises the whole model on every round, and a large
    submission would otherwise pay its whole size per round."""
    from onnx import AttributeProto, TensorProto

    def strip(t):
        if int(np.prod(t.dims)) > _FOLD_MAX_ELEMENTS:
            for f in ("raw_data", "float_data", "int32_data", "int64_data", "double_data", "uint64_data",
                      "string_data", "external_data"):
                t.ClearField(f)
            t.data_location = TensorProto.DEFAULT

    for init in graph.initializer:
        strip(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.type == AttributeProto.TENSOR:
                    strip(a.t)
        for sub in _subgraphs(node):
            _strip_weights(sub)


def _static_model(model, batch: int):
    """A copy of `model` (weights dropped, see `_strip_weights`) with the batch axis bound to
    `batch` and every shape that is fixed at that size resolved: shape inference (with data
    propagation), then folding of the shape arithmetic exporters emit -- `Shape -> Gather ->
    Concat -> Equal -> Where -> Expand` and the like, where plain data propagation stops -- to
    a fixpoint.  What stays symbolic is genuinely data-dependent (`NonZero`, a trip count read
    from the data), and the counter refuses it."""
    import onnx
    from onnx import shape_inference
    m = onnx.ModelProto()
    m.CopyFrom(model)
    _strip_weights(m.graph)
    _bind_batch(m.graph, batch)
    opsets = {imp.domain: imp.version for imp in m.opset_import}
    for _ in range(_FOLD_ROUNDS):
        m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
        if not _fold_graph(m.graph, {}, {}, opsets):
            return m
    return shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)


# -- FLOPs -------------------------------------------------------------------------------


def _label(node) -> str:
    return f"{node.op_type} {node.name or (node.output[0] if node.output else '?')!r}"


def _trip_count(node, known: dict) -> int:
    """A `Loop`'s trip count when it is a constant; `ValidationError` otherwise.  A loop may
    stop earlier through its condition -- the constant `M` is then an upper bound, never an
    undercount."""
    m_name = node.input[0] if node.input else ""
    cond = node.input[1] if len(node.input) > 1 else ""
    if cond and cond in known and not bool(np.asarray(known[cond]).reshape(-1)[0]):
        return 0                                                     # never entered
    if not m_name:
        raise ValidationError(f"{_label(node)} has no max_trip_count: it runs until its condition turns false, "
                              f"a dynamic trip count that cannot be counted -- give the loop a constant trip count")
    if m_name not in known:
        raise ValidationError(f"{_label(node)}: max_trip_count {m_name!r} is not a constant -- a dynamic trip count "
                              f"cannot be counted; make it an initializer or a Constant")
    return max(0, int(np.asarray(known[m_name]).reshape(-1)[0]))


def _scan_length(node, shapes: dict) -> int:
    n_scan = int(_attr(node, "num_scan_inputs", 0) or 0)
    if n_scan <= 0 or n_scan > len(node.input):
        raise ValidationError(f"{_label(node)}: num_scan_inputs {n_scan} does not fit its {len(node.input)} inputs")
    axes = list(_attr(node, "scan_input_axes", []) or [0] * n_scan)
    lengths = set()
    for name, ax in zip(node.input[len(node.input) - n_scan:], axes):
        s = shapes.get(name)
        if not s:
            raise ValidationError(f"{_label(node)}: scan input {name!r} has no static shape after shape inference "
                                  f"at B = 1 -- the sequence length cannot be counted; make the shape static")
        lengths.add(int(s[ax % len(s)]))
    if len(lengths) != 1:
        raise ValidationError(f"{_label(node)}: scan inputs disagree on the sequence length {sorted(lengths)}")
    return lengths.pop()


def _count_graph(graph, outer_shapes: dict, outer_known: dict) -> tuple:
    """(FLOPs, per-op breakdown) of one pass over `graph`, every shape static."""
    shapes, known = _scope(graph, outer_shapes, outer_known)

    def static(name, node):
        s = shapes.get(name)
        if s is None:
            raise ValidationError(
                f"{_label(node)} uses tensor {name!r} whose shape is not static after shape inference at B = 1 "
                f"(a symbolic dimension is left, or none was inferred), so its FLOPs cannot be counted -- make the "
                f"shape static: fixed sizes on every axis but B, no data-dependent shapes (NonZero, boolean masks, "
                f"a trip count read from the data)")
        return s

    total, by_op = 0, {}
    for node in graph.node:
        op = node.op_type
        if op in _CONTROL_FLOW:
            if op == "If":
                costs = [_count_graph(g, shapes, known) for g in _subgraphs(node)]
                f, by = max(costs, key=lambda c: c[0]) if costs else (0, {})
            else:
                body = _attr(node, "body")
                if body is None:
                    raise ValidationError(f"{_label(node)} has no body")
                f_body, by = _count_graph(body, shapes, known)
                reps = _trip_count(node, known) if op == "Loop" else _scan_length(node, shapes)
                f, by = f_body * reps, {k: v * reps for k, v in by.items()}
            total += f
            for k, v in by.items():
                by_op[k] = by_op.get(k, 0) + v
            continue
        if op in _ZERO_COST or not node.output or not node.output[0]:
            f = 0
        else:
            n_out = int(np.prod(static(node.output[0], node)))
            ins = [static(i, node) for i in node.input if i]
            if op in ("Conv", "ConvTranspose"):
                x, w = ins[0], ins[1]
                kk = int(np.prod(w[2:]))
                if op == "Conv":                                     # weight (Cout, Cin/g, k...): Cin/g * k*k MACs per output element
                    f = 2 * n_out * w[1] * kk
                else:                                                # weight (Cin, Cout/g, k...): each input element reaches Cout/g * k*k outputs
                    f = 2 * int(np.prod(x)) * w[1] * kk
            elif op == "MatMul":
                f = 2 * n_out * ins[0][-1]
            elif op == "Gemm":
                a = ins[0]
                f = 2 * n_out * (a[0] if _attr(node, "transA", 0) else a[-1])
            else:
                f = n_out                                            # one FLOP per output element
        total += f
        by_op[op] = by_op.get(op, 0) + f
    return total, by_op


def flops_of(model, batch: int = 1) -> tuple:
    """FLOPs of one call at `B = batch` from static shapes, plus the per-op breakdown.
    Subgraphs are counted (`Loop` x constant trip count, `Scan` x sequence length, the
    costlier `If` branch).  Raises `ValidationError` -- naming the tensor or the loop --
    when a counted operator's tensor has no static shape after inference and folding, or a
    loop's trip count is not a constant: the graph is refused rather than under-counted."""
    m = _static_model(model, batch)
    return _count_graph(m.graph, {}, {})


# --------------------------------------------------------------------------------
# Dynamic: shapes, determinism, statelessness, the execution contract
# --------------------------------------------------------------------------------


def _probe_inputs(sub, name: str, B: int, rng) -> dict:
    spec = sub.spec[name]["inputs"]
    feed = {}
    for i in sub.sessions[name].get_inputs():
        shp = list(spec[i.name])
        shp[0] = B
        if i.name in ("frame",):
            x = rng.uniform(0, 1, shp)
        elif i.name in ("actions", "action_prev"):
            x = rng.uniform(-1, 1, shp)
        elif i.name in ("level",):
            x = np.full(shp, 0.0)
        elif i.name in ("step",):
            x = np.full(shp, 1.0)
        elif i.name in ("z", "z_final", "grid", "grids"):
            x = np.tanh(rng.standard_normal(shp))
        else:                                                        # ctx: a real one if possible
            x = rng.uniform(-1, 1, shp)
        feed[i.name] = x.astype(np.float32)
    return feed


def dynamic_checks(sub, rep: Report, seed: int = 0, tol: float = 1e-4):
    rng = np.random.default_rng(seed)
    spec = sub.spec
    for name in sub.manifest.graphs:
        # -- output shapes on a real-shaped probe -----------------------------------
        feed2 = _probe_inputs(sub, name, 2, rng)
        out2 = sub.run(name, **feed2)
        for oname, oshape in spec[name]["outputs"].items():
            if oname not in out2:
                rep.fail(f"{name}: output {oname!r} missing")
                continue
            got = out2[oname].shape
            want = (2,) + tuple(oshape[1:])
            if tuple(got) != want:
                rep.fail(f"{name}: output {oname} has shape {got}, contract wants {want}")
            if not np.all(np.isfinite(out2[oname])):
                rep.fail(f"{name}: output {oname} contains NaN/Inf")
        # -- determinism -----------------------------------------------------------------
        again = sub.run(name, **feed2)
        for oname in out2:
            if oname in again and not np.allclose(out2[oname], again[oname], atol=tol, rtol=tol):
                rep.fail(f"{name}: re-running a recorded call changed {oname} beyond {tol} (not deterministic)")
        # -- statelessness across the batch ----------------------------------------------
        feed1 = {k: v[1:2] for k, v in feed2.items()}
        out1 = sub.run(name, **feed1)
        for oname in out2:
            if oname in out1 and not np.allclose(out2[oname][1], out1[oname][0], atol=tol, rtol=tol):
                rep.fail(f"{name}: row 1 of a batch of 2 differs from the single-row call on {oname} "
                         f"(rows must not interact)")
    # -- the execution contract on decoded frames -----------------------------------
    if "visual_decode" in sub.sessions:
        z = CT.initial_latent(seed, 1, sub.manifest.chunk_size)
        fr = sub.run("visual_decode", z_final=z)["frames"]
        lo, hi = float(fr.min()), float(fr.max())
        rep.numbers["decode_probe_range"] = [lo, hi]
        if lo < -0.05 or hi > 1.05:
            rep.fail(f"visual_decode: frames outside [0, 1] ({lo:.3f}, {hi:.3f}) on a probe latent")
    if "act" in sub.sessions:
        feed = _probe_inputs(sub, "act", 4, rng)
        a = sub.run("act", **feed)["action"]
        if np.abs(a).max() > 1.0 + 1e-6:
            rep.fail(f"act: action outside [-1, 1] ({np.abs(a).max():.3f})")


# --------------------------------------------------------------------------------
# The whole check
# --------------------------------------------------------------------------------


def validate(path: str, run_dynamic: bool = True, provider: str = "CPUExecutionProvider") -> Report:
    from . import runner as RN
    rep = Report()
    mp = os.path.join(path, "manifest.json")
    if not os.path.exists(mp):
        rep.fail("manifest.json missing")
        return rep
    with open(mp) as f:
        m = CT.Manifest.from_json(json.load(f))
    for b in m.check():
        rep.fail("manifest: " + b)
    rep.numbers["manifest"] = m.to_json()
    rep.numbers["ctx_bytes"] = m.ctx_bytes()
    models = {}
    for name in m.graphs:
        gp = os.path.join(path, "graphs", f"{name}.onnx")
        if not os.path.exists(gp):
            rep.fail(f"graph {name} missing (every submission carries {list(m.graphs)})")
            continue
        try:
            models[name] = _load(gp)
        except ValidationError as e:
            rep.fail(f"{name}: {e}")
    if rep.errors:
        return rep
    for name, model in models.items():
        rep.numbers.setdefault("ops", {})[name] = check_operators(model, name, rep)
    try:
        total, per = parameter_count(models)
    except ValidationError as e:
        rep.fail(f"parameters: {e}")
        return rep
    rep.numbers["parameters"] = dict(total_shared_once=total, per_graph=per)
    if total > CT.PARAMS_MAX:
        rep.fail(f"{total:,} parameters, cap {CT.PARAMS_MAX:,}")
    flops = {}
    for name, model in models.items():
        try:
            f, by = flops_of(model, 1)
            flops[name] = f
        except ValidationError as e:                                 # the graph cannot be counted: rejected, not under-counted
            rep.fail(f"{name}: {e}")
        except Exception as e:                                       # noqa: BLE001
            rep.warn(f"{name}: FLOP counting failed ({type(e).__name__}: {e})")
    rep.numbers["flops_per_call_b1"] = flops
    K, N = m.chunk_size, m.prediction_steps
    # FLOPs are counted and reported, not capped: compute is bounded by the wall-clock limits
    if "predict_step" in flops:
        rep.numbers["predict_flops_per_frame"] = flops["predict_step"] * N / K
    if "visual_decode" in flops:
        rep.numbers["decode_flops_per_frame"] = flops["visual_decode"] / K
    if all(k in flops for k in ("visual_encode", "observe", "act")):
        rep.numbers["control_flops_per_step"] = flops["visual_encode"] + flops["observe"] + flops["act"]
    if run_dynamic:
        try:
            sub = RN.Submission(path, provider=provider)
        except Exception as e:                                       # noqa: BLE001
            rep.fail(f"could not open the submission: {type(e).__name__}: {e}")
            return rep
        for name, sess in sub.sessions.items():
            want_in = sub.spec[name]["inputs"]
            got_in = {i.name: tuple(i.shape) for i in sess.get_inputs()}
            extra = set(got_in) - set(want_in)
            if extra:
                rep.fail(f"{name}: unexpected inputs {sorted(extra)}")
            for k, got in got_in.items():
                if k in want_in:
                    shp = want_in[k]
                    if len(got) != len(shp) or any(a != b for a, b in zip(got[1:], shp[1:])):
                        rep.fail(f"{name}: input {k} has shape {got}, contract wants {shp}")
            want_out = sub.spec[name]["outputs"]
            got_out = {o.name for o in sess.get_outputs()}
            if got_out != set(want_out):
                rep.fail(f"{name}: outputs {sorted(got_out)}, contract wants {sorted(want_out)}")
        if not rep.errors:
            dynamic_checks(sub, rep)
        rep.numbers["timing_ms"] = sub.timing()
        enc = rep.numbers["timing_ms"].get("encode")
        if enc and enc["p95_ms"] > CT.ENCODE_SECONDS_MAX * 1e3:
            msg = (f"encode {enc['p95_ms'] / 1e3:.2f} s per call (p95) against the "
                   f"{CT.ENCODE_SECONDS_MAX:.0f} s limit under the pinned runtime")
            if sub.provider == "CUDAExecutionProvider":
                rep.fail(msg)
            else:
                rep.warn(msg + f" -- measured on {sub.provider}; the official check runs on the pinned runtime")
    return rep


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Track 3 graph validator: the same checks intake runs")
    ap.add_argument("submission")
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = validate(a.submission, run_dynamic=not a.static_only)
    if a.json:
        print(json.dumps(rep.to_json(), indent=1))
    else:
        print("OK" if rep.ok else "REJECTED")
        for e in rep.errors:
            print("  error:", e)
        for w in rep.warnings:
            print("  warning:", w)
        n = rep.numbers
        if "parameters" in n:
            print(f"  parameters: {n['parameters']['total_shared_once']:,} (shared tensors once)")
        for k, v in n.get("flops_per_call_b1", {}).items():
            print(f"  {k}: {v / 1e6:.1f} MFLOPs per call at B=1")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
