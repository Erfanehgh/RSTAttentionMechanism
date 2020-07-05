class Node(object):
    isLeaf = True
    isRoot = False
    leftChild = 0
    rightChild = 0
    child = "left"
    nodeHierarchy = "N"
    nodeRelation = "null"
    vector = []
    delta = []

    # The class "constructor" - It's actually an initializer
    def __init__(self, isLeaf, isRoot, leftChild, rightChild, child, hierarchy, relation, vector, delta):
        self.isLeaf = isLeaf
        self.isRoot = isRoot
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.child = child
        self.vector = vector
        self.nodeHierarchy = hierarchy
        self.nodeRelation = relation
        self.delta = delta

