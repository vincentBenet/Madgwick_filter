from qgis._core import QgsExpressionContext, QgsExpression, QgsFieldProxyModel


def eval_expression_widget(layer, widget):
    expression_str = widget.expression()
    return eval_expression(layer, expression_str)


def eval_expression(layer, expression_str):
    expression = QgsExpression(expression_str)
    context = QgsExpressionContext()
    result = []
    for i, feature in enumerate(layer.getFeatures()):
        context.setFeature(feature)
        value = expression.evaluate(context)
        result.append(value)
    return result


def filter_expression(widget, type_output):
    if type_output in [float, int]:
        type_widget = QgsFieldProxyModel.Numeric
    elif type_output in [str]:
        type_widget = QgsFieldProxyModel.String
    elif type_output in [bool]:
        type_widget = QgsFieldProxyModel.Boolean
    else:
        raise Exception
    widget.setFilters(type_widget)
