from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsTextItem,QGraphicsRectItem,QGraphicsPolygonItem,QGraphicsItem
from PyQt6.QtCore import QRectF, QPointF,Qt,QLineF
from PyQt6.QtGui import QPen, QColor,QPolygonF,QFont
import math

class GraphicsNode(QGraphicsRectItem):
    def __init__(self, node_id, x, y, main_app=None, parent=None):
        self.node_id = node_id
        self.main_app = main_app

        label = self.main_app.graph.nodes[node_id].get('label', str(node_id))
        self.text_item = QGraphicsTextItem(label)
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))  # White text
        is_kutahya = label.lower().startswith('kütah')

        if is_kutahya:
            font_size = 50
            padding = 30
            default_width = 400
            pen_thickness = 6
            rect_dimensions = [-40, -40, 400, 200]
        else:
            font_size = 18
            padding = 20
            default_width = 160
            pen_thickness = 4
            rect_dimensions = [-30, -30, 160, 80]

        font = QFont("Roboto", font_size)
        font.setBold(True)
        self.text_item.setFont(font)  # Increase font size to 16

        text_width = self.text_item.boundingRect().width() + padding
        node_width = max(default_width, text_width)
        rect_dimensions[2] = node_width
        super().__init__(QRectF(*tuple(rect_dimensions)))


        self.setPos(QPointF(x, y))
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setBrush(QColor(0x1F, 0x14, 0x5D))
        self.setPen(QPen(QColor(0, 0, 0), pen_thickness))  # Thicker border

        self.text_item.setParentItem(self)
        # Calculate the center points of both items
        rect_center = self.boundingRect().center()
        text_center = self.text_item.boundingRect().center()

        # Calculate the position to center the text
        text_pos = rect_center - text_center

        self.text_item.setPos(text_pos)


    def update_position(self):
        start_pos = self.start_node.scenePos()
        end_pos = self.end_node.scenePos()
        self.setLine(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.update_connected_edges()
        return super().itemChange(change, value)

    def update_connected_edges(self):
        for edge in self.main_app.graph.edges:
            if edge.start_node == self or edge.end_node == self:
                edge.update_position()


    def mousePressEvent(self, event):

        if event.button() != Qt.MouseButton.LeftButton:
            return  #

        super().mousePressEvent(event)
        scene_pos = self.scenePos()
        # print(f"Clicked on Node {self.node_id} at Scene Coordinate: ({scene_pos.x()}, {scene_pos.y()})")
        if event.modifiers() != Qt.KeyboardModifier.ControlModifier:
            # If Ctrl is not pressed, clear the selection
            self.main_app.selected_nodes.clear()
            self.main_app.selected_nodes.append(self)
        else:
            # Ctrl is pressed, toggle selection
            if self in self.main_app.selected_nodes:
                self.main_app.selected_nodes.remove(self)
            else:
                self.main_app.selected_nodes.append(self)

    def mouseReleaseEvent(self, event):
        if self.main_app.context_menu_triggered:
            self.main_app.context_menu_triggered = False
            return
        super().mouseReleaseEvent(event)
        x, y = self.pos().x(), self.pos().y()
        self.main_app.graph.nodes[self.node_id]['pos'] = (x, y)

class GraphicsEdge(QGraphicsLineItem):
    def __init__(self, start_node, end_node, main_app=None):
        self.start_node = start_node
        self.end_node = end_node

        start_pos, end_pos = self.adjust_edge_endpoints()
        super().__init__(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())

        self.setPen(QPen(QColor(0, 0, 0, 255), 8))

        # Arrowhead
        arrow_head = QPolygonF()
        arrow_head.append(QPointF(0, 0))
        arrow_head.append(QPointF(5, 15))
        arrow_head.append(QPointF(-5, 15))
        self.arrow_item = QGraphicsPolygonItem(arrow_head, self)
        self.arrow_item.setBrush(QColor(0, 0, 0, 255))
        self.adjustArrow()

    def update_position(self):
        start_pos, end_pos = self.adjust_edge_endpoints()
        self.setLine(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())

    def adjust_edge_endpoints(self):
        start_rect = self.start_node.sceneBoundingRect()
        end_rect = self.end_node.sceneBoundingRect()

        start_pos = self.start_node.scenePos() + QPointF(start_rect.width() / 2, start_rect.height() / 2)
        end_pos = self.end_node.scenePos() + QPointF(end_rect.width() / 2, end_rect.height() / 2)

        adjusted_start = self.get_intersection_point(end_rect, end_pos, start_pos)
        adjusted_end = self.get_intersection_point(start_rect, start_pos, end_pos)

        return adjusted_start, adjusted_end

    def get_intersection_point(self, rect, ref_point, point):
        intersections = []
        edges = [
            QLineF(rect.topLeft(), rect.topRight()),
            QLineF(rect.topRight(), rect.bottomRight()),
            QLineF(rect.bottomRight(), rect.bottomLeft()),
            QLineF(rect.bottomLeft(), rect.topLeft())
        ]

        line = QLineF(ref_point, point)

        for edge in edges:
            intersect_type, intersection_point = edge.intersects(line)
            if intersect_type == QLineF.IntersectionType.BoundedIntersection:
                intersections.append(intersection_point)

        # Find the closest intersection point to ref_point
        if intersections:
            return min(intersections, key=lambda p: QLineF(p, ref_point).length())
        else:
            # This shouldn't happen, but just in case.
            return point


    def angleBetween(self, p1, p2):
        delta_x = p2.x() - p1.x()
        delta_y = p2.y() - p1.y()
        radians = math.atan2(delta_y, delta_x)
        return math.degrees(radians)

    def adjustArrow(self):
        line = self.line()
        dx = line.x2() - line.x1()
        dy = line.y2() - line.y1()
        angle = math.degrees(math.atan2(dy, dx))

        length = math.sqrt(dx**2 + dy**2)

        # Set a larger arrow offset if needed, to position the arrowhead closer to the endpoint
        arrow_offset = 13*length/16   # Adjust as needed for the arrow position
        self.arrow_item.setRotation(angle - 90)  # -90 to align the arrow with the line

        # Create a larger arrowhead by adjusting the coordinates of the polygon
        arrow_head = QPolygonF()
        arrow_head.append(QPointF(0, 0))
        arrow_head.append(QPointF(15, 40))  # Adjust these values to make the arrowhead larger
        arrow_head.append(QPointF(-15, 40))  # Adjust these values to make the arrowhead larger
        self.arrow_item.setPolygon(arrow_head)  # Set the new polygon to arrow_item

        direction = self.normalized(line.p2() - line.p1())
        arrow_pos = line.p2() - direction * arrow_offset  # Move towards the line end
        self.arrow_item.setPos(arrow_pos)

    def setLine(self, *args):
        super().setLine(*args)
        self.adjustArrow()

    def normalized(self, vector):
        length = math.sqrt(vector.x() ** 2 + vector.y() ** 2)
        if length == 0:
            return QPointF(0, 0)
        return QPointF(vector.x() / length, vector.y() / length)


