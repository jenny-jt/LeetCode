
##########1431. Kids With the Greatest Number of Candies ##########
# Given the array candies and the integer extraCandies, 
# where candies[i] represents the number of candies that the ith kid has.

# For each kid check if there is a way to distribute extraCandies among the kids 
# such that he or she can have the greatest number of candies among them. 
# Notice that multiple kids can have the greatest number of candies.

class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        """given list of int and int of extra candies, return True/False if can distribute extra candies so that kid can have greatest # of candies"""
        # dict of kid/index: num candies
        # find max num candies
        # true: max - original candiese <= extra
        # false: max - original candiese > extra
        
        # initialize empty array & dict
        d = {}
        result = []
        
        # find max # candies
        max_candies = max(candies)

        # create dictionary with kid as key and value as num_candies
        for i, num_candies in enumerate(candies):
            d[i] = num_candies
            if max_candies - num_candies <= extraCandies:
                result.append(True)
            else:
                result.append(False)

        return result

##########455. Assign cookies ##########

# Assume you are an awesome parent and want to give your children some cookies. 
# But, you should give each child at most one cookie.

# Each child i has a greed factor g[i], which is the minimum size of a cookie 
# that the child will be content with; and each cookie j has a size s[j]. 
# If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will 
# be content. Your goal is to maximize the number of your content children and output the maximum number.


class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        """given list of child_content and list of cookie_size
            return num of maximized content children"""

        g.sort()
        s.sort()
        
        counter, i, j = 0, 0, 0
        while j < len(g) and i < len(s):
            if s[i] >= g[j]:
                counter, i, j = counter+1, i+1, j+1
            else:
                i +=1

        return counter

##########13. Roman to Integer ##########
# Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

# Symbol       Value
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000
# For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

# Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

# I can be placed before V (5) and X (10) to make 4 and 9. 
# X can be placed before L (50) and C (100) to make 40 and 90. 
# C can be placed before D (500) and M (1000) to make 400 and 900.
# Given a roman numeral, convert it to an integer.


def romanToInt(self, s: str) -> int:
    """given roman numeral, return the integer version"""
    result = 0
    c_dict = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}

    for i in range(len(s) - 1):
        if c_dict[s[i]] < c_dict[s[i+1]]:
            result -= c_dict[s[i]]
        else:
            result += c_dict[s[i]]
    # don't forget to convert and add last roman numeral
    result += c_dict[s[len(s)-1]]

    return result

# Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

# An input string is valid if:

# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
 
##########20. Valid Parentheses ##########
def isValid(self, s: str) -> bool:
    """given string s, return True if valid, False if not"""
    d = {"(": ")", "[": "]", "{": "}"}
    stack = []

    for parens in s:
        if parens in d:
            stack.append(parens)
        else:
            if stack and parens == d[stack[-1]]:
                stack.pop()
            else:
                return False

    if stack:
        return False
    else:
        return True

##########155. Min Stack ##########
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x: int) -> None:
        """ add item to end"""
        self.stack.append(x)

    def pop(self) -> None:
        """remove item from end"""
        self.stack.pop()

    def top(self) -> int:
        """return item at the end"""
        return self.stack[-1]

    def getMin(self) -> int:
        """return minimum item"""
        return min(self.stack)


##########206. Reverse Linked List ##########
def reverseList(self, head: ListNode) -> ListNode:
    """reverse a SLL"""
    prev = None
    curr = head

    while curr != None:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp

    return prev

########## 21. Merge Two Sorted Lists ##########
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

####### iterative solution #########


def mergeTwoListsIter(self, l1: ListNode, l2: ListNode) -> ListNode:
    """ given 2 sorted linked lists, return new sorted list as a LL"""
    result = ListNode()
    curr = result

    # until reach tail of both l1 and l2
    while l1 or l2:
        # if l1 empty, return l2 and done
        if not l1:
            curr.next = l2
            break
        # if l2 empty, return l1 and done
        if not l2:
            curr.next = l1
            break
        # if first item of l1 is smaller or equal, append that one
        if l1.val <= l2.val:
            curr.next = ListNode(l1.val)
            l1 = l1.next
        # if first item of l2 is smaller, append that one
        else:
            curr.next = ListNode(l2.val)
            l2 = l2.next
        curr = curr.next

    return result.next

###### recursive solution ######


def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    """ given 2 sorted LL, return new sorted list as a LL"""
    if not l1:
        return l2
    elif not l2:
        return l1
    elif l1.val <= l2.val:
        l1.next = self.mergeTwoLists(l1.next, l2)
        print("l1", l1)
        return l1
    else:
        l2.next = self.mergeTwoLists(l2.next, l1)
        print("l2", l2)
        return l2


########## 1137. N-th Tribonacci Number ##########
# The Tribonacci sequence Tn is defined as follows: 
# T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.
# Given n, return the value of Tn.
class Solution:
    cache = {}
    def tribonacci(self, n: int) -> int:
        # base case T_0 = 0     
        # base case T_1 = 1
        # base case T_2 = 1
#         T_3 = t2 + t1 + t0 = 1 + 1 + 0 = 2
#         t(n) = t(n-1) + t(n-2) + t(n-3)

        if n == 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 1

        if n not in self.cache:
            self.cache[n] = self.tribonacci(n-1) + self.tribonacci(n-2) + self.tribonacci(n-3)
        return self.cache[n]


########## 104. Maximum Depth of Binary Tree ##########
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# first 
def maxDepth(self, root: TreeNode) -> int:
    """given root of binary tree, return int max depth """

    def traverse(node):
        # base case
        if not node:
            return 0
        # max depth of left nodes
        l_max = traverse(node.left)
        # max depth of right nodes
        r_max = traverse(node.right)
        # add one to account for depth from root
        return max(l_max, r_max) + 1

    return traverse(root)

# second
def maxDepth(self, root: TreeNode) -> int:
    """given root of binary tree, return int max depth """
    
    def traverse(node, depth):
        # base case
        if not node:
            return depth
        # keep going L until hit leaf node
        left = traverse(node.left, depth + 1)
        # keep going R until hit leaf node
        right = traverse(node.right, depth + 1)
        return max(left, right)

    return traverse(root, 0)

# root = [3,9,20,null,null,15,7]
# node = 3 
# traverse(3) = max(L, R) = max(2,3)
# left = traverse(9, 1) = max(2,2) = 2                         right = traverse(20, 1) = max(3,3) = 3
# L = traverse(None,2)  R = traverse(None,2)      L= traverse(15,2) = max(3,3) = 3          R=traverse(7,2) = max(3,3) = 3 
# 2                       2                       L/R = traverse(none,3)                    L/R = traverse(none,3)


########## 938. Range of BST ##########
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        """given root, return int sum of nodes with values between high and low"""
        # BST: all L children are less than R children
        # if root out of range, then children on L will be out of range, no need to keep checking
        # base case: single node with value in range
        res = []

        def traverse(node):
            if node:
                # if val in range, add val
                if low <= node.val <= high:
                    res.append(node.val)
                # if val greater than low, can explore L branch. otherwise stop
                if node.val > low:
                    traverse(node.left)
                # if val less than high, can explorer R branch. otherwise stop
                if node.val < high:
                    traverse(node.right)
            else:
                res.append(0)

        traverse(root)        
        return sum(res)

########## 617. Merge Two Binary Trees  ##########
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        """given 2 binary trees, sum up overlapping nodes and copy non-overlapping nodes"""
        # start at root of both trees, modify tree 1
        # compare(if both not null, sum = new node.val)
        # preorder/in order traversal and compare
        # how set L and R nodes, how traverse new tree

        # preorder(node):
        #     node
        #     preorder(node.left)
        #     preorder(node.right)
        if not t1:
            return t2
        if not t2:
            return t1
        # change t1 val to be sum of t1 and t2
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)

        # return root of modified
        return t1


########## 897. Increasing Order Search Tree  ##########
    def increasingBST(self, root: TreeNode) -> TreeNode:
        """given root of BST, return tree with min node on L with only R children"""
        # in order traversal, save to q ideally
        # return smallest node
        # set each R child to next node

        # inoorder(node):
        #     node.left
        #     node
        #     node.right
        q = []

        def inorder(node):
            if node:        
                inorder(node.left)
                q.append(node.val)
                inorder(node.right)

        inorder(root)
        print(q)

        curr = root_new = TreeNode(None)
        for i in range(len(q)):
            curr.right = TreeNode(q[i]) 
            curr = curr.right

        return root_new.right

########## 700. Search in Binary Search Tree  ##########
def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    """given root of BST and value, return subtree with root of value
        return null if no node with value
    """
    # DFS to use recursion
    # base case: root node has val, return root node
    def traverse(node):
        if not node:
            return None
        if node:
            if node.val == val:
                return node
            left = traverse(node.left)
            if left:
                return left
            else:
                return traverse(node.right)

    return traverse(root)


########## 559. Maximum Depth of N-ary Tree  ##########
def maxDepth(self, root: 'Node') -> int:
    """given root of tree, return int max depth"""
    # base case
    if not root:
        return 0

    def traverse(node, depth):
        if not node.children:
            return depth
        # find max depth of traverse(each child node, depth + 1)
        else:  
            n_depth = max([traverse(child, depth+1) for child in node.children])
            return n_depth

    return traverse(root, 1)

# Example case #1 [1,null,3,2,4,null,5,6]
#     node 1
#     node.c = [3, 2, 4]
#     n_depth = max(traverse(3,2), traverse(2,2), traverse(4,2)) 
#         traverse(3,2)             2               2
#         node 3
#         node.c = [5, 6]
#         n_depth = max(traverse(5,3), traverse(6,3))
#         traverse(5,3)         traverse(6,3)
#           3                       3
# Example case of [1]
#     node 1
#     node.c = []
#     n_level = traverse(none, 1)
#         depth = 1

########## 965. Univalued Binary Tree  ##########
    def isUnivalTree(self, root: TreeNode) -> bool:
        """given root of tree, return True if univalued, False if not"""
        # base case: not root, return True
        if not root:
            return True
        # base case: only root, return True
        if not root.left and not root.right:
            return True

        def traverse(node):
            # if node val not equal to root, return False
            print("node val", node.val, "root val", root.val)
            if node.val != root.val:
                return False
            # if leaf node and not false:
            if not node.left and not node.right:
                return True
            # if only right child
            if not node.left:
                return traverse(node.right)
            # if only left child
            if not node.right:
                return traverse(node.left)
            # if both children
            else:
                return traverse(node.left) and traverse(node.right)

        return traverse(root)


########## 669. Trim a Binary Search Tree  ##########
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        """given root of BST, trim tree so all nodes in low-high, return root of edited tree"""
        
        def traverse(node):
            # base case
            if not node:
                return
            # if node is too low, check if right node in range
            if node.val < low:
                return traverse(node.right)
            # if node is too high, check if left node in range
            if node.val > high:
                return traverse(node.left)
            # if node in range
            else:
                # check if left subtree in range. if it is a leaf, it will return at next call
                node.left = traverse(node.left)
                # check if right subtree in range
                node.right = traverse(node.right)
                return node
        
        return traverse(root)
        
# root = [3,0,4,null,2,null,null,1], range = 1-3

# traverse(3)
# node.left = traverse(0)									node.right = traverse(4)				return 3
# traverse(0)												traverse(4)							
# traverse(2)												traverse(None)
# node.left = traverse(1)  node.right = traverse(None) return 2	return
# traverse(1)			  return
# node.left = traverse(None)
# node.right = traverse(None)
# Return 1

########## 344. Reverse String  ##########
# reverse string in place, using no other data structure
def reverseString(self, s: List[str]) -> None:
    """
    Do not return anything, modify s in-place instead.
    """
    # cannot initiate another data structure
    # use the swap syntax in python

    for i in range(len(s)//2):
        s[i], s[len(s)-1-i] = s[len(s)-1-i], s[i]


########## 136. Single Number  ##########
def singleNumber(self, nums: List[int]) -> int:
    """given array return int that is single"""
    
    d = Counter(nums)
    
    for key, value in d.items():
        if d[key] == 1:
            return key


########## 387. First Unique Character in a String  ##########
# if not first non-repeating char, return -1
def firstUniqChar(self, s: str) -> int:
    """given string, return index of first non-repeating char"""
    for i in range(len(s)):
        if s.count(s[i]) == 1:
            return i
    return -1


########## 230. Kth Smallest Element in a BST  ##########
def kthSmallest(self, root: TreeNode, k: int) -> int:
    """given BST, return kth smallest element in it"""
    # initialize list
    # in order traversal to add nodes to list
    # index into list (k-1)
    l = []
    
    def inordert(node):
        if node:
            inordert(node.left)
            l.append(node.val)
            inordert(node.right)

    inordert(root)
    return l[k-1]


######### 238. Product of Array Except Self ##############
# need to optimize runtime
def productExceptSelf(self, nums: List[int]) -> List[int]:
    """given array of int > 1, return array of products for all indices except each index"""
    
    # for i in nums, pop off and save as a variable
    # multiply nums and insert that into array
    # insert back into nums and pop off next one
    result = []
    
    for i in range(len(nums)):
        temp = nums.pop(i)
        product = reduce((lambda x, y: x * y), nums)
        result.append(product)
        nums.insert(i, temp)
    
    return result

# using arrays to store left and right sided product
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """given array of int > 1, return array of products for all indices except each index"""
        L = [1]*len(nums)
        R = [1]*len(nums)

        for i in range(0, len(nums)):
            if i == 0:
                L[i] = 1 
            else:
                L[i] = L[i-1]*nums[i-1]
            
        R[len(nums)-1] = 1
        for i in reversed(range(len(nums)-1)):
            R[i] = R[i+1]*nums[i+1]
        print(L, R)
        result = [L[i]*R[i] for i in range(len(nums))]
        
        return result


######### 507. Perfect Number ##############
def checkPerfectNumber(self, num: int) -> bool:
    """given int n, return true if n is a perfect number, otherwise false"""
    
    # num = 6
    # 1, 2, 3 = 1 + 2 + 3 = 6; True
    
    # num = 496
    # 1, 2, 248, 4, 124, 
    
    ans = set()

    for digit in range(1, int(sqrt(num)) + 1):
        # integer will equal float if not decimal value after it
        if int(num/digit) == num/digit:
            ans.add(digit)
            ans.add(int(num/digit))  # float
        
    if sum(ans) - num == num:
        return True
    else:
        return False


######### 283. Move Zeroes ##############
def moveZeroes(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    
    for num in nums:
        if num == 0:
            nums.remove(num)
            nums.append(num)
            
    return nums


######### 830. Positions of Large Groups ##############
def largeGroupPositions(self, s: str) -> List[List[int]]:
    """given string s, return array of indices of large group"""
    # 2 pointers, i and j
    l = []
    i = 0
    for j in range(i+1, len(s)):
        if j == len(s) - 1 or s[j] != s[j+1]:
            if j-i+1 >= 3:
                l.append([i,j])
            i = j+1
                
    return l
        

######### 242. Valid Anagram ##############
def isAnagram(self, s: str, t: str) -> bool:
    """given 2 strings s and t, return true if anagrams, else False"""
    if Counter(s) == Counter(t):
        return True
    else:
        return False


######### 125. Valid Palindrome ##############
def isPalindrome(self, s: str) -> bool:
    """given string s, return True if palindrome, False if not"""
    punct = string.punctuation
    s = s.lower()
    new_s = ''
    
    if len(s) == 0:
        return True
    else:
        for char in s:
            if char not in punct:
                new_s += char
    new_l = new_s.split()
    l = ''.join(new_l)
    print(l)
        
    x = len(l)//2
    # use a stack maybe
    # first half, reverse it and compare to second half
    rev_s = l[:x]
    print("rev", rev_s)
    
    if len(l) % 2 == 0:
        sec_s = l[x:]
    else:
        sec_s = l[x + 1:]
        
    print("sec", sec_s)
        
    if rev_s[::-1] == sec_s:
        return True
    else:
        return False

