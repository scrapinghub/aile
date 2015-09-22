import selenium
import selenium.webdriver

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def equal_delta(a, b, delta):
    return (a - delta <= b) and (b <= a + delta)


class BBox(object):
    def __init__(self):
        self._empty = True

        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

    def wrap(self, element):
        if self._empty:
            self.x1 = element.x
            self.y1 = element.y
            self.x2 = element.x + element.width
            self.y2 = element.y + element.height
        else:
            self.x1 = min(self.x1, element.x)
            self.x2 = max(self.x2, element.x + element.width)
            self.y1 = min(self.y1, element.y)
            self.y2 = max(self.y2, element.y + element.height)

        self._empty = False

    def contains(self, other):
        if self._empty or other._empty:
            return False

        return (self.x1 <= other.x1 and
                self.x2 >= other.x2 and
                self.y1 <= other.y1 and
                self.y2 >= other.y2)

    def halign(self, other, margin=5):
        return (equal_delta(self.y1, other.y1, margin) and
                equal_delta(self.y2, other.y2, margin))

    def valign(self, other, margin=5):
        return (equal_delta(self.x1, other.x1, margin) and
                equal_delta(self.x2, other.x2, margin))


class DOM(object):
    class Element(object):
        def __init__(self, parent=None, children=None):
            self.parent = parent
            self.children = children or []

    def __init__(self, browser, flat=False):
        def make_element(node, parent):
            element = DOM.Element(parent=parent)
            for k, v in node.rect.iteritems():
                setattr(element, k, v)
            element.x = max(0, element.x)
            element.y = max(0, element.y)
            return element

        root_node = browser.find_elements_by_xpath('*')[0]
        if flat:
            self.root = make_element(root_node, None)
            for child in root_node.find_elements_by_xpath('//*'):
                self.root.children.append(make_element(child, self.root))
        else:
            def fill(node, parent=None):
                element = make_element(node, parent)
                for child in node.find_elements_by_xpath('*'):
                    element.children.append(fill(child, parent=element))
                return element

            self.root = fill(root_node)

    def draw(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            ax.invert_yaxis()

        def _draw(element, bbox):
            ax.add_patch(
                patches.Rectangle(
                    (element.x, element.y),
                    element.width,
                    element.height,
                    fill=False
                )
            )

            bbox.wrap(element)
            for child in element.children:
                _draw(child, bbox)

        bbox = BBox()
        _draw(self.root, bbox)

        ax.set_xlim(bbox.x1, bbox.x2)
        ax.set_ylim(bbox.y1, bbox.y2)


def get_dom(url):
    browser = selenium.webdriver.Firefox()
    browser.get(url)
    dom = DOM(browser, flat=True)
    browser.close()
    return dom


if __name__ == '__main__':
    dom = get_dom('http://edition.cnn.com/')
    dom.draw()
    plt.show()
