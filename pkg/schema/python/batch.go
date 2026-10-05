package python

import (
	"fmt"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"

	"github.com/replicate/cog/pkg/schema"
)

// Limits mirrored from python/cog/_batch.py.
const (
	maxBatchSizeLimit     = 1024
	maxBatchWaitMsLimit   = 60_000
	defaultBatchMaxWaitMs = 5.0
)

const batchSignatureHint = "@batch run() must take exactly one parameter annotated list[Item], " +
	"where Item is a cog.BaseModel subclass defined in the same file, and return list[Output]"

// functionBatchConfig returns the @cog.batch configuration of a run() method,
// or nil when the method is not batched.
func functionBatchConfig(node *sitter.Node, source []byte, imports *schema.ImportContext) (*schema.BatchInfo, error) {
	decorated := decoratedFunctionNode(node)
	if decorated == nil {
		return nil, nil
	}
	for _, child := range NamedChildren(decorated) {
		if child.Type() != "decorator" {
			continue
		}
		expr := decoratorExpression(child)
		if expr == nil || !expressionIsCogDecorator(expr, source, imports, "batch") {
			continue
		}
		return parseBatchDecorator(child, source)
	}
	return nil, nil
}

func parseBatchDecorator(node *sitter.Node, source []byte) (*schema.BatchInfo, error) {
	call := decoratorCall(node)
	if call == nil {
		return nil, schema.WrapError(schema.ErrUnsupportedType, "@batch must be called with arguments: @batch(max_batch_size=...)", nil)
	}
	info := &schema.BatchInfo{MaxWaitMs: defaultBatchMaxWaitMs}
	haveSize := false
	if argList := callArgumentList(call); argList != nil {
		for _, arg := range NamedChildren(argList) {
			if arg.Type() != "keyword_argument" {
				return nil, schema.WrapError(schema.ErrUnsupportedType, "@batch only supports keyword arguments max_batch_size=... and max_wait_ms=...", nil)
			}
			nameNode := arg.ChildByFieldName("name")
			valueNode := arg.ChildByFieldName("value")
			if nameNode == nil || valueNode == nil {
				return nil, schema.WrapError(schema.ErrUnsupportedType, "@batch only supports literal keyword arguments", nil)
			}
			val, ok := parseDefaultValue(valueNode, source)
			switch Content(nameNode, source) {
			case "max_batch_size":
				if !ok || val.Kind != schema.DefaultInt {
					return nil, schema.WrapError(schema.ErrUnsupportedType, "@batch max_batch_size must be an integer literal", nil)
				}
				if val.Int < 1 || val.Int > maxBatchSizeLimit {
					return nil, schema.WrapError(schema.ErrUnsupportedType, fmt.Sprintf("@batch max_batch_size must be between 1 and %d", maxBatchSizeLimit), nil)
				}
				info.MaxBatchSize = int(val.Int)
				haveSize = true
			case "max_wait_ms":
				var wait float64
				switch {
				case ok && val.Kind == schema.DefaultInt:
					wait = float64(val.Int)
				case ok && val.Kind == schema.DefaultFloat:
					wait = val.Float
				default:
					return nil, schema.WrapError(schema.ErrUnsupportedType, "@batch max_wait_ms must be a number literal", nil)
				}
				if wait < 0 || wait > maxBatchWaitMsLimit {
					return nil, schema.WrapError(schema.ErrUnsupportedType, fmt.Sprintf("@batch max_wait_ms must be between 0 and %d", maxBatchWaitMsLimit), nil)
				}
				info.MaxWaitMs = wait
			default:
				return nil, schema.WrapError(schema.ErrUnsupportedType, fmt.Sprintf("@batch got unexpected argument %q", Content(nameNode, source)), nil)
			}
		}
	}
	if !haveSize {
		return nil, schema.WrapError(schema.ErrUnsupportedType, "@batch requires max_batch_size=...", nil)
	}
	return info, nil
}

// expressionIsCogDecorator reports whether a decorator expression refers to
// cog.<name> (via `import cog` or `from cog import <name>`, including aliases).
func expressionIsCogDecorator(node *sitter.Node, source []byte, imports *schema.ImportContext, name string) bool {
	switch node.Type() {
	case "attribute":
		module, attr, ok := strings.Cut(Content(node, source), ".")
		if !ok || attr != name {
			return false
		}
		entry, ok := imports.Names.Get(module)
		return ok && entry.Module == "cog" && entry.Original == "cog"
	case "identifier":
		entry, ok := imports.Names.Get(Content(node, source))
		return ok && entry.Module == "cog" && entry.Original == name
	default:
		return false
	}
}

// listElementType returns X for list[X] / List[X] annotations.
func listElementType(ann schema.TypeAnnotation) (schema.TypeAnnotation, bool) {
	if ann.Kind != schema.TypeAnnotGeneric || len(ann.Args) != 1 {
		return schema.TypeAnnotation{}, false
	}
	switch ann.Name {
	case "list", "List", "typing.List":
		return ann.Args[0], true
	}
	return schema.TypeAnnotation{}, false
}

// extractBatchInputs derives the per-request input schema of a batched
// run(self, inputs: list[Item]) from the fields of Item.
func extractBatchInputs(paramsNode *sitter.Node, source []byte, ctx *inputParseContext, file *pythonFileContext) (*schema.OrderedMap[string, schema.InputField], error) {
	var params []*sitter.Node
	for _, child := range NamedChildren(paramsNode) {
		if child.Type() == "identifier" && Content(child, source) == "self" {
			continue
		}
		params = append(params, child)
	}
	if len(params) != 1 || params[0].Type() != "typed_parameter" {
		return nil, schema.WrapError(schema.ErrUnsupportedType, batchSignatureHint, nil)
	}
	_, typeNode := typedParameterParts(params[0], source)
	if typeNode == nil {
		return nil, schema.WrapError(schema.ErrUnsupportedType, batchSignatureHint, nil)
	}
	ann, err := parseTypeAnnotation(typeNode, source)
	if err != nil {
		return nil, err
	}
	elem, ok := listElementType(ann)
	if !ok || elem.Kind != schema.TypeAnnotSimple {
		return nil, schema.WrapError(schema.ErrUnsupportedType, batchSignatureHint, nil)
	}
	classNode := findLocalClass(file.root, source, elem.Name)
	if classNode == nil || !InheritsFromBaseModel(classNode, source, file.imports) {
		return nil, schema.WrapError(schema.ErrUnsupportedType,
			fmt.Sprintf("@batch input type %q must be a cog.BaseModel subclass defined in the same file as run()", elem.Name), nil)
	}
	return extractModelInputs(classNode, source, ctx)
}

func findLocalClass(root *sitter.Node, source []byte, name string) *sitter.Node {
	for _, child := range NamedChildren(root) {
		classNode := UnwrapClass(child)
		if classNode == nil {
			continue
		}
		if nameNode := classNode.ChildByFieldName("name"); nameNode != nil && Content(nameNode, source) == name {
			return classNode
		}
	}
	return nil
}

// extractModelInputs turns annotated class fields (`name: type = default`)
// into input fields, with the same Input()/default handling as run() params.
func extractModelInputs(classNode *sitter.Node, source []byte, ctx *inputParseContext) (*schema.OrderedMap[string, schema.InputField], error) {
	inputs := schema.NewOrderedMap[string, schema.InputField]()
	body := classNode.ChildByFieldName("body")
	if body == nil {
		return inputs, nil
	}
	order := 0
	for _, stmt := range NamedChildren(body) {
		node := stmt
		if stmt.Type() == "expression_statement" && stmt.NamedChildCount() == 1 {
			node = stmt.NamedChild(0)
		}
		if node.Type() != "assignment" {
			continue
		}
		left := node.ChildByFieldName("left")
		typeNode := node.ChildByFieldName("type")
		if left == nil || left.Type() != "identifier" || typeNode == nil {
			// Unannotated class attributes are not dataclass fields.
			continue
		}
		field, err := buildInputField(Content(left, source), typeNode, node.ChildByFieldName("right"), source, order, ctx)
		if err != nil {
			return nil, err
		}
		inputs.Set(field.Name, field)
		order++
	}
	return inputs, nil
}

// batchOutputType resolves the per-item output schema from a list[Output]
// return annotation.
func batchOutputType(returnAnn schema.TypeAnnotation, file *pythonFileContext) (schema.SchemaType, error) {
	elem, ok := listElementType(returnAnn)
	if !ok {
		return schema.SchemaType{}, schema.WrapError(schema.ErrUnsupportedType, batchSignatureHint, nil)
	}
	return schema.ResolveSchemaType(elem, file.imports, file.modelClasses)
}
