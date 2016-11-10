# coding:utf-8
import xml.dom.minidom


filename = "segments/ES2002a."
person = ["A", "B", "C", "D"]

dom = xml.dom.minidom.parse(filename+'A.segments.xml')

print dom.documentElement.tagName
for node in dom.documentElement.childNodes:
    if node.nodeType == node.ELEMENT_NODE:
        print '  ' + node.tagName
        for node2 in node.childNodes:
            if node2.nodeType == node2.ELEMENT_NODE:
                print '    ' + node2.tagName
                for node3 in node2.childNodes:
                    if node3.nodeType == node3.TEXT_NODE:
                        print '      ' + node3.data
