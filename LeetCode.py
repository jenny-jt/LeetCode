
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
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        """given root of binary tree, return int max depth """
        return self.traverse(root)

    def traverse(self, root):
        # base case
        if not root:
            return 0

        left_max = self.traverse(root.left)
        right_max = self.traverse(root.right)
        
        # return larger of left and right
        return max(left_max, right_max) + 1

        