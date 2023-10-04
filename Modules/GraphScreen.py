from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt, QPoint

class PannableView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super(PannableView, self).__init__(*args, **kwargs)
        self._is_panning = False
        self._mouse_last_pos = QPoint()
        self.setSceneRect(-2500, -2500, 5000, 5000)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoomInFactor = 1.25
            zoomOutFactor = 1 / zoomInFactor

            oldPos = self.mapToScene(event.position().toPoint())

            if event.angleDelta().y() > 0:
                zoomFactor = zoomInFactor
            else:
                zoomFactor = zoomOutFactor
            self.scale(zoomFactor, zoomFactor)

            newPos = self.mapToScene(event.position().toPoint())

            delta = newPos - oldPos
            self.translate(delta.x(), delta.y())
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        # If the left button is pressed while no item is selected, we start panning
        if event.button() == Qt.MouseButton.LeftButton and not self.itemAt(event.pos()):
            self._is_panning = True
            self._mouse_last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)  # change cursor appearance to indicate panning
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            # Calculate how much the mouse moved since the last mouseMoveEvent
            delta = self.mapToScene(event.pos()) - self.mapToScene(self._mouse_last_pos)
            # Move the view accordingly
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._mouse_last_pos = event.pos()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)  # restore cursor appearance
        else:
            super().mouseReleaseEvent(event)


