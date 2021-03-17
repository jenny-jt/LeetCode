
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


########## 206. Reverse Linked List ##########
# iteration without curr
# https://www.youtube.com/watch?v=NhapasNIKuQ&t=239s&ab_channel=NickWhite
def reverseList(self, head: ListNode) -> ListNode:
    """reverse a SLL"""
    _next = None
    prev = None

    while head:
        _next = head.next
        head.next = prev
        prev = head
        head = _next

    return prev


# iteration with curr
def reverseList(self, head: ListNode) -> ListNode:
    """reverse a SLL"""
    prev = None
    curr = head

    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp

    return prev


# recursion
def reverseList(self, head: ListNode) -> ListNode:
    """reverse a SLL"""
    # base cases
    if not head or not head.next:
        return head
    # make list smaller each time until reach a reversed list (base case)
    rev_head = reverseList(head.next)
    # append myself to the reversed list (grows as we go back up the stack)
    head.next.next = head
    # prevents cycle from forming
    head.next = None
    # keep track of the head of the reversed LL
    return rev_head

# recursion #2
def reverseList(self, head: ListNode) -> ListNode:
    if not head or not head.next:
        return head

    next_node = head.next
    # prevent cycles of having head.next of new_head defined differently each time
    head.next = None
    next_node.next = head
    new_head = self.reverseList(next_node)

    return new_head

# recursion #3
def recursive(head, previous):
    if head is None:
        return previous
    result = recursive(head.next, head)
    head.next = previous
    return result
    

return recursive(head, None)

1 - 2 - 3 - 4 - 5 - none

rL(1) next=2, next.next=1, head.next would be 2 but is now None
rL(2) next=3, next.next=2
rL(3) next=4, next.next=3
rL(4) next=5, next.next=4
rL(5) = 5


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
# recursive, first 
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

# recursive, second
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

# recursive, 3rd
def maxDepth(self, root: TreeNode) -> int:
    """given root of binary tree, return int max depth """
    count = 0

    def dfs(node, count):
        if not node:
            return count
        return max(dfs(node.left, count +1), dfs(node.right, count +1))

    return dfs(root, count)


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
# splitting word in half and comparing to other half
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

# using 2 pointers
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
    
    for i in range(len(l)//2):
        j = len(l) - i - 1
        if l[i] != l[j]:
            return False
    
    return True


######### 997. Find the Town Judge ##############
def findJudge(self, N: int, trust: List[List[int]]) -> int:
    """given trust, return label of town judge, else -1"""
    # keep trust count
    # town judge has trust count N-1
    
    # only one person in the town, return that person
    if len(trust) == 0 and N == 1:
        return 1
    
    tc = [0] * (N+1)
    
    for item in trust:
        tc[item[0]] -= 1
        tc[item[1]] += 1
    
    for count in tc:
        if count == N-1:
            return tc.index(count)
    
    return -1


######### 200. Number of Islands ##############
def numIslands(self, grid: List[List[str]]) -> int:
    """given grid of land and water, return int number of islands"""
    # island: top bottom left and right are surrounded by water
    
    count = 0
    
    def dfs(grid, row, col):
        # stop points(zero, out of grid)
        if row < 0 or col < 0:
            return
        if row > len(grid) -1 or col > len(grid[0]) -1:
            return
        if grid[row][col] == "0":
            return
        
        # zero it out
        grid[row][col] = "0"
        
        # call dfs (top, bottom, left, right)
        dfs(grid, row-1,col)
        dfs(grid, row+1, col)
        dfs(grid, row, col-1)
        dfs(grid, row, col+1)
    
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "1":
                dfs(grid, row, col)
                count += 1
                
    return count


######### 417. Pacific Atlantic Water Flow ##############
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
    """given mxn matrix, return array of coordinates where pacific -> atlantic"""
    if not matrix: 
        return []
    
    def dfs(i,j,matrix,explored,prev):
        m,n = len(matrix),len(matrix[0])
        # if out of bounds or already seen
        if i < 0 or i >= m or j < 0 or j >= n or (i,j) in explored:
            return
        # only add if water will flow to ocean (height must be greater than previous)
        if matrix[i][j] < prev:
            return
        # add new, valid point to ocean
        explored.add((i,j))
        dfs(i-1,j,matrix,explored,matrix[i][j]) #up
        dfs(i+1,j,matrix,explored,matrix[i][j]) #down
        dfs(i,j-1,matrix,explored,matrix[i][j]) #left
        dfs(i,j+1,matrix,explored,matrix[i][j]) #right      
    
    pacific,atlantic = set(),set()
    m,n = len(matrix),len(matrix[0])
    for i in range(n):
        dfs(0,i,matrix,pacific,-1)
        dfs(m-1,i,matrix,atlantic,-1)
    for i in range(m):
        dfs(i,0,matrix,pacific,-1)
        dfs(i,n-1,matrix,atlantic,-1)
    
    # return intersection of both oceans
    return list(pacific&atlantic)


######### 141. Linked List Cycle ##############
# second attempt
def hasCycle(self, head: ListNode) -> bool:
    """given LL, return True if cycle present, otherwise False"""
    
    # if there is one node or head is None, then return False
    if not head or not head.next:
        return False
    # fast pointer and slow pointer, and seeing if they equal each other
    fast = slow = head
    # traverse faster w fast; end of LL, fast will be None
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            return True

# first attempt
def hasCycle(self, head: ListNode) -> bool:
    """given LL, return True if cycle present, otherwise False"""
    
    if not head:
        return False
    
    curr2 = head
    curr1 = head.next
    
    while curr1 != curr2:
        # if either are none, have reached end of list
        if not curr1 or not curr1.next:
            return False
        curr1 = curr1.next.next
        curr2 = curr2.next

    return True


######### 141. Reverse Linked List ##############
def reverseList(self, head: ListNode) -> ListNode:
    """reverse a SLL"""
    prev = None
    curr = head
    
    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    
    return prev


######### 11. Container With Most Water ##############
# brute force, need to optimize
def maxArea(self, height: List[int]) -> int:
    """given array of ints, return max value of area of ints"""
    # brute force: calculate all areas and return max
    # width = i2 - i1
    # height = min(num1, num2)
    # area = height * width
    
    i, area = 0, 0
    
    while i < len(height) -1:
        for j in range(len(height)):
            h = min(height[i], height[j])
            w = abs(j - i)
            area = max(area, h*w)
        i += 1
    
    return area

# 2 pointers
def maxArea(self, height: List[int]) -> int:
    """given array of ints, return max value of area of ints"""
    # 2 pointers, at opposite ends of array
    # advance pointer with shorter height
    # width = j - i
    # height = min(num1, num2)
    # area = height * width
    
    i = len(height) -1
    area, j = 0, 0
    
    while i > 0 and j < len(height):
        h = min(height[i], height[j])
        w = abs(j - i)
        area = max(area, h*w)
        
        if height[i] <= height[j]:
            i -= 1
        else:
            j += 1
    
    return area


######### 21. Merge Two Sorted Lists ##############
# iterative solution
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    """ given 2 sorted LL, return new sorted list as a LL"""
    
    res = head = ListNode(0)
    
    while l1 and l2:
        if l1.val < l2.val:
            res.next = l1
            l1 = l1.next
        else:
            res.next = l2
            l2 = l2.next
        res = res.next
        
    if l2:
        res.next = l2
        # append rest of l2
    if l1:
        res.next = l1
        # append rest of l1
    
    return head.next

# recursive
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        """ given 2 sorted LL, return new sorted list as a LL"""
        
        if not l1:
            return l2
        if not l2:
            return l1
        
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
        
#         l1  l2
#         1   1
#         2   1
#         2   3
#         4   3
#         4   4
#         none 4

#         return l1 = 1
#         l1.next = m(2,1) = 1
#         l2.next = m(2,3) = 2
#         l1.next = m(4,3) = 3
#         l2.next = m(4,4) = 4
#         l1.next = m(none,4) = 4


######### 739. Daily Temperatures ##############
# brute force, need to optimize
def dailyTemperatures(self, T: List[int]) -> List[int]:
    """given array of int, return array of int of days until warmer, othersiwe 0"""
    # brute force, iterate through i and rest of list
    # return i and j of max(listj-listi) if >0, otherwise return 0
    res = []
    
    for i in range(len(T)):
        # set max_T to 0 for each comparison
        max_T = 0
        for j in range(i+1,len(T)):
            # print("i", i, "j", j)
            if T[j] - T[i] > max_T:
                max_T = j - i
            # once hit any hotter temp, break out of loop and go to next i
            if max_T > 0:
                break
        res.append(max_T)
        
    return res

# using stack
def dailyTemperatures(self, T: List[int]) -> List[int]:
    """given array of int, return array of int of days until warmer, otherwise 0"""
    stack = []
    res = [0] * len(T)
    
    # using stack, push indices of higher temps on, from the back
    for i in range(len(T)-1, -1, -1):
        while stack and T[i] >= T[stack[-1]]:
            stack.pop()
        if stack:
            res[i] = stack[-1] - i
        stack.append(i)
    
    return res
                

######### 1. Two Sum ##############
# brute force
def twoSum(self, nums: List[int], target: int) -> List[int]:
    """given array of int and target int, return indices of 2 int that sum up to target"""
    # brute force: find sum for each pair, if num is smaller than target
    # use 2 pointers
    
    # if num < target, find if num-target in array and return that index
    
    for i in range(len(nums)):
        other_num = target - nums[i]
    
        if other_num in nums:
            j = nums.index(other_num)
            if j != i:
                return [i, j]

# hash table
def twoSum(self, nums: List[int], target: int) -> List[int]:
    """given array of int and target int, return indices of 2 int that sum up to target"""
    # hash table
    
    d = {}
    
    for i in range(len(nums)):
        # checks if num in hash_table keys
        if nums[i] in d:
            # if yes, return the index of other_num, i
            return([d[nums[i]], i])
        else:
            # set key of other_num to value of index
            d[target - nums[i]] = i


######### 49. Group Anagrams ##############
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    """given array of strings, return array of array of strings, clustered by anagram"""
    
    # anagrams have same sorted string
    
    d = defaultdict(list)

    # generate dict with key word and value counter 
    for word in strs:
        s_str = ''.join(sorted(word))
        d[s_str].append(word)
    
    res = list(d.values())
    
    return res


######### 994. Rotting Oranges ##############
def orangesRotting(self, grid: List[List[int]]) -> int:
    """given m x n matrix, return int minutes until no cell has a fresh orange, otherwise -1"""
    
    # 0: empty cell
    # 1: fresh orange
    # 2: rotten orange
    # every min, adjacent fresh orange becomes rotten
    # how check if any oranges are unreached
    # bfs to find minutes
    
    fresh, minute = 0, 0
    q = []
        
    # find count of fresh oranges and add rotten with coordinates to q
    for row in range(len(grid)):
        for col in range(len(grid[0])):       
            # count fresh oranges
            if grid[row][col] == 1:
                fresh += 1
            # find location of rotten ones and append them to q
            if grid[row][col] == 2:
                q.append((row,col, minute))
                
    while q:
        row, col, minute = q.pop(0)
        
        if grid[row][col] == 2:
            for r,c in [(row, col+1), (row-1,col), (row,col-1), (row+1,col)]:
                if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1:
                    grid[r][c] = 2
                    fresh -= 1
                    q.append((r,c, minute + 1))

    # if leftover fresh oranges:
    if fresh:
        return -1
    
    return minute



######### 73. Set Matrix Zeroes ##############
def setZeroes(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    q = []
    m = len(matrix)
    n = len(matrix[0])
    
    def helper(matrix, row, col):
        # process cells
        # set row and column to 0
        for r in range(m):
            if matrix[r][col] != 0:
                matrix[r][col] = 0
        for c in range(n):
            if matrix[row][c] != 0:
                matrix[row][c] = 0
        
    # walk through matrix
    for row in range(m):
        for col in range(n):
            if matrix[row][col] == 0: 
                q.append((row, col))
    
    while q:
        row, col = q.pop()
        helper(matrix, row, col)


######### 232. Implement Queue using Stacks ##############
class MyQueue:
    """implement q using 2 stacks"""
def __init__(self):
    """
    Initialize your data structure here.
    """
    self.stack1 = []
    self.stack2 = []

def push(self, x: int) -> None:
    """
    Push element x to the back of queue.
    """
    self.stack1.append(x)

def pop(self) -> int:
    """
    Removes the element from in front of queue and returns that element.
    """
    n = len(self.stack1) - 1
    # move all items except first from stack1 to stack2
    for i in range(n):
        self.stack2.append(self.stack1.pop())
    # return the original first item
    res = self.stack1.pop()
    # move the remaining items of the first list
    for i in range(n):
        self.stack1.append(self.stack2.pop())
        
    return res

def peek(self) -> int:
    """
    Get the front element.
    """
    n = len(self.stack1) - 1
    
    for i in range(n):
        self.stack2.append(self.stack1.pop())
        
    res = self.stack1[0]
    
    for i in range(n):
        self.stack1.append(self.stack2.pop())
    return res

def empty(self) -> bool:
    """
    Returns whether the queue is empty.
    """
    return len(self.stack1) == 0


######### 415. Add Strings ##############
def addStrings(self, num1: str, num2: str) -> str:
    """given 2 nums as strings, return sum"""
    res = []
    extra = 0
    
    # find length of strings
    l1 = len(num1)
    l2 = len(num2)
    
    # make both strings the same length
    if l1 < l2:
        num1 = num1.zfill(l2)
    if l1 > l2:
        num2 = num2.zfill(l1)
    
    # make list of tuples of digits from l1 and l2
    char_list = list(zip(num1, num2))
    print(char_list)
    
    for item in char_list[::-1]:
        digit1, digit2 = item
        sum_ = int(digit1) + int(digit2) + extra
        extra = sum_ // 10
        res.append(str(sum_ % 10))
    
    if extra:
        res.append(str(extra))
        
    return ''.join(res[::-1])


######### 78. Subsets ##############

# iteration, faster
def subsets(self, nums: List[int]) -> List[List[int]]:
    """given array of int, return set of all possible subsets"""

    output = [[]]
    
    for num in nums:
        # [] + [1] = [1] so each time a num is added, output will have single num with combos
        output += [item + [num] for item in output]
        # print(output)
        
    return output

# recursion, slower
def subsets(self, nums: List[int]) -> List[List[int]]:
    """given array of int, return set of all possible subsets"""
    if not nums:
        return [[]]
    
    others = self.subsets(nums[1:])
    # print(others)
    
    # add on nums[0]
    return [[nums[0]] + seq for seq in others] + others


######### 62. Unique Paths ##############
# helper function
def uniquePaths(self, m: int, n: int) -> int:
    """dp solution"""
    # make griid
    res = [[0] * n for r in range(m)]
    # helper to fill in grid                
    def helper(r, c, res):
        if r == 0 or c == 0:
            res[r][c] = 1
        else:
            res[r][c] = res[r-1][c] + res[r][c-1]
    # fill in grid
    for r in range(m):
        for c in range(n):
            helper(r,c,res)

    # return index so m-1, n-1
    return res[m-1][n-1]

# recursion
cache = {}

def uniquePaths(self, m: int, n: int) -> int:

    if m == 1 or n == 1:
        return 1
    
    if (m,n) in cache:
        return cache[(m,n)]

    cache[(m,n)] = self.uniquePaths(m-1, n) + self.uniquePaths(m, n-1)
        
    return cache[(m,n)]

# 3,3
# cache[3,3] = self(2,3) + self(3,2) = 3 + 3 = 6
# cache[3,2] = self(3,1) + self(2,2) = 1 + 2 = 3
# cache[3,1] = 1

# cache[2,2] = self(1,2) + self(2,1) = 1 + 1 = 2
# cache[1,2] = 1
# cache[2,1] = 1

# cache[2,3] = self(1,3) + self(2,2) = 1 + 2 = 3
# cache[1,3] = 1

# cache[2,2] = self(1,2) + self(2,1) = 1 + 1 = 2
# cache[1,2] = 1
# cache[2,1] = 1

# 1 2 3
# 2 x x
# 3 x x

# dynamic programming
def uniquePaths(self, m: int, n: int) -> int:
    """dp solution"""
    
    # initialize matrix to values of 1, cover the borders with only 1 possible path
    res = [[1] * n for i in range(m)]

    # pass through matrix and fill in values inside borders
    for r in range(1,m):
        for c in range(1,n):
            res[r][c] = res[r-1][c] + res[r][c-1]

    # return last row and last column
    return res[m-1][n-1]


######### 53. Maximum Subarray ##############

def maxSubArray(self, nums: List[int]) -> int:
    """given array of int, return sum of contiguous subarray which has the largest sum"""
    
    # initialize sums array to store max sum for each index of nums
    sums = [0] * len(nums)
    
    # loop through nums
    for i in range(len(nums)):
        # sums for that index is either num or num+prev sum
        sums[i] = max(sums[i-1]+nums[i], nums[i])
    
    # return max of sums array
    return max(sums)


######### 300. Longest Increasing Subsequence ##############
# dp
def lengthOfLIS(self, nums: List[int]) -> int:
    """given array of int, return int len of longest increasing subsequence"""
    # time complexity: O(n^2), space complexity: O(n)

    N = len(nums)
    dp = [1] * N
    
    for i in range(N):
        # for all nums up to i, compare if i can be appended to j
        for j in range(i):
            # can be appended if nums[i] > nums[j]
            if nums[i] > nums[j]:
                # max of either 1 (i itself) or length at j + 1 (because i appended)
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# recursion

def lengthOfLIS(self, nums: List[int]) -> int:
    """given array of int, return int len of longest increasing subsequence"""

    cache = {}
    
    def dfs(nums, i, prev):
        # base case (covers empty nums, nums of one element)
        if i == len(nums):
            return 0

        # if this num is greater than prev num, choose greater of 1+length @ prev, or length @ curr
        if nums[i] > prev:
            return max(dfs(nums, i+1, prev), 1 + dfs(nums, i+1, nums[i]))
        # if not, then return length @ prev
        return dfs(nums, i+1, prev)
    
    return dfs(nums, 0, float('-inf'))

    # [0,1,0]
    # len = 3
    # i   prev    nums[i]    
    # 0   -inf    0       max(dfs(1,-inf), 1+dfs(1,0)) = max(1,2) = 2
    # 1   -inf    1       max(dfs(2,-inf), 1+dfs(2,1)) = max(0,1) = 1
    # 1   0       1       max(dfs(2,0), 1+dfs(2,1)) = max(0,1) = 1
    # 2   0       0       dfs(3,0) = 0
    # 2   1       0       dfs(3,0) = 0
    # 2   -ind    0       dfs(3,0) = 0
    # 3   0       3 == 3  return 0


######### 1502. Can Make Arithmetic Progression From Sequence ##############
def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
    """given array of numbers, return boolean True if array can be a progression, otherwise False"""
    
    # sort array
    # small amount of work: diff between consecutive nums
    # if diff not equal to previous diff, return False

    diff_set = set()
    
    array = sorted(arr)
    
    for i in range(len(array)-1):
        diff = array[i+1] - array[i]
        diff_set.add(diff)
    
    if len(diff_set) != 1:
        return False
    
    return True
    

######### 322. Coin Change ##############
def coinChange(self, coins: List[int], amount: int) -> int:
    """given coins and total, return int fewest coins to make amt or -1 if not able to make amt"""
    
    dp = [amount+1] * (amount+1)
    
    dp[0] = 0
    
    for i in range(amount+1):
        for j in range(len(coins)):
            if coins[j] <= i:
                dp[i] = min(dp[i], 1+dp[i-coins[j]])    
    
    if dp[i] < amount + 1:
        return dp[i]
    else:
        return -1

# recursion*** does not work
def coinChange(coins, amount):
    """given coins and total, return int fewest coins to make amt or -1 if not able to make amt"""
    if amount == 0:
        return 0

    # loop through coins and subtract coin from amount, find min num of needed coins
    for coin in coins:
        min_c = coin
        if amount - coin >= 0:
            min_c = min(min_c, coinChange(amount-coin, coins) + 1)
        print("min", min_c)
    
    return min_c
 

######### 146. LRU Cache ##############
# iterative
def removeDuplicates(self, nums: List[int]) -> int:
    """given array of ints, remove duplicates,
        return len(new_array) and array without duplicates"""

    j = 1
    for i in range(1,len(nums)):
        if nums[i] != nums[i-1]:
            nums[j] = nums[i]
            j += 1
    # del rest of nums from the back
    nums = nums[:j]

    return len(nums)

    # nums = [0,0,1,1,1,2,2,3,3,4]

    # I.  J. nums[I] nums[I-1]
    # 1 1 0 0
    # 2 1 1 0 nums[1] -> nums[2]
    # [0,1,1,1,1,2,2,3,3,4]
    # 3 2 1 1
    # 4 2 1 1
    # 5 2 2 1
    # [0,1,2,1,1,2,2,3,3,4]
    # 6 3 2 2 
    # 7 3 3 2 
    # [0,1,2,3,1,2,2,3,3,4]
    # 8 4 3 3
    # 9 4 4 3 
    # [0,1,2,3,4,2,2,3,3,4]

# recursive
def removeDuplicates(self, nums: List[int]) -> int:
    """given sorted array of ints, remove duplicates,
        return len(new_array)"""

    # base case
    if not nums:
        return 0
    if len(nums) == 1:
        return 1
        
    def dfs(nums, i):
        if i == len(nums)-1:
            return len(nums)
        if nums[i] != nums[i+1]:
            return dfs(nums, i+1)
        nums.pop(i+1)
        return dfs(nums, i)
    
    return dfs(nums, 0)


######### 207. Course Schedule #############
from collections import defaultdict
# second attempt
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    """given list of courses and list of prereq, return boolean True if can take all courses"""
    if not prerequisites:
        return True
    
    stack = []
    visited = set()
    courses = defaultdict(list)
    incoming = defaultdict(int)
    # first dependent on second
    # loop through prereqs and make dicts of incoming and of courses with prereq
    for course, pre in prerequisites:
        # can decrease incoming dependency for each course that has the prereq that we "take"
        courses[pre].append(course)
        incoming[course] += 1
        
    # "take" courses without prereqs if not already taken
    for i in range(numCourses):
        if incoming[i] == 0:
            stack.append(i)
            visited.add(i)
    # update prereqs based on those in res already
    count = 0
    while stack:
        take = stack.pop()
        # decrease incoming for course with the prereq that was "taken"
        for course in courses[take]:
            incoming[course] -= 1
            # if course can now be taken and has not already been taken, add it to stack
            # code still runs if not use visited condition, actually faster- why? should be checking a set
            if incoming[course] == 0 and course not in visited:
                stack.append(course)
        # will check if we "took" the same number of courses as numcourses
        count += 1
    
    return count == numCourses
            

def canFinish(numCourses, prerequisites):
    """given list of courses and list of prereq
    return boolean True if can take all courses
    >>> canFinish(2, [[1,0]])
    True
    """
    # cycle: return false
    
    if not prerequisites:
        return True
    
    course_list = defaultdict(list)
    indegrees = {n: 0 for n in range(numCourses)}
    
    # assemble dict of pre:[courses], and for each pre, increase course indegrees value
    for course, pre in prerequisites:
        course_list[pre].append(course)
        indegrees[course] += 1
        
    # take classes that don't have prereqs
    take = [course for course in indegrees if indegrees[course] == 0]
    # print("take", take)
    
    # if no courses with indegree of 0, can't take any classes
    if not take:
        return False
    
    # initialize set to keep track of taken courses
    complete = set()
    
    while take:
        to_take = take.pop()
        complete.add(to_take)
        
        if to_take in course_list:
            for course in course_list[to_take]:
                indegrees[course] -= 1
                if indegrees[course] == 0:
                    take.append(course)
                    
    if len(complete) == numCourses:               
        return True


######### 207. Course Schedule 2 ##############
from collections import defaultdict

# second attempt, using default dict and list comprehension
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """given list of courses and list of prereq, return boolean True if can take all courses"""
    if not prerequisites:
        return [course for course in range(numCourses)]

    stack, complete = [], []
    courses = defaultdict(list)
    incoming = defaultdict(int)
    # first dependent on second
    # loop through prereqs and make dicts of incoming and of key(pre): value(courses with that prereq)
    for course, pre in prerequisites:
        # can decrease incoming dependency for each course that has the prereq that we "take"
        courses[pre].append(course)
        incoming[course] += 1

    # "take" courses without prereqs if not already taken
    stack = [course for course in range(numCourses) if incoming[course] == 0]
    
    # update prereqs based on taken classes
    while stack:
        take = stack.pop()
        # add taken class to ans
        complete.append(take)
        # decrease incoming for course with the prereq that was "taken"
        for course in courses[take]:
            incoming[course] -= 1
            # if course can now be taken and has not already been taken, add it to stack
            # code still runs if not use visited condition, actually faster- why? should be checking a set
            if incoming[course] == 0:
                stack.append(course)
                
    # will check if we "took" the same number of courses as numcourses
    if len(complete) == numCourses:
        return complete
    return []
        

def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

    if not prerequisites:
        return [n for n in range(numCourses)]

    course_list = defaultdict(list)
    indegrees = {n: 0 for n in range(numCourses)}

    # assemble dict of pre:[courses], and for each pre, increase course indegrees value
    for course, pre in prerequisites:
        course_list[pre].append(course)
        indegrees[course] += 1

    # take classes that don't have prereqs
    take = [course for course in indegrees if indegrees[course] == 0]
    # print("take", take)
    
    # initialize set to keep track of taken courses
    complete = []

    while take:
        
        to_take = take.pop()
        complete.append(to_take)
        # print("complete", complete)

        if to_take in course_list:
            for course in course_list[to_take]:
                indegrees[course] -= 1
                if indegrees[course] == 0:
                    take.append(course)
                    
    if len(complete) == numCourses:
        return complete
    else:
        return []


######### 15. 3Sum ##############
# recursive, need to optimize
def threeSum(self, nums: List[int]) -> List[List[int]]:
    """given nums, return all unique triplets where a + b + c = 0"""
    # check ab, increment c
    # then keep a, increment b and c
    # then increment a

    result = []
    nums.sort()
    
    if len(nums) < 3:
        return result
    
    if len(nums) == 3 and (nums[0] + nums[1] + nums[2] == 0):
        return [nums]
    
    def dfs(nums, a, b, c, result):
        # if nums add up to 0, append to result
        sum_ = nums[a] + nums[b] + nums[c]
        
        if sum_ == 0:
            if sorted([nums[a], nums[b], nums[c]]) not in result:
                result.append(sorted([nums[a], nums[b], nums[c]]))
        if a == len(nums)-3:
            return result
        if b == len(nums)-2:
            return dfs(nums, a+1, a+2, a+3, result)
        if c == len(nums)-1:
            return dfs(nums, a, b+1, b+2, result)

        return dfs(nums, a, b, c+1, result)
    
    return dfs(nums,0,1,2, result)

    # nums = [-1,0,1,2,-1,-4]
    #          0 1 2 3  4  5
    # len(nums) = 6
    # a   b   c   nums[a] nums[b] nums[c] sum result
    # 0   1   2   -1      0       1       0   [[-1,0,1]]
    # 0   1   3   -1      0       2       1   [[-1,0,1]]
    # 0   1   4   -1      0       -1      -2   [[-1,0,1]]
    # 0   1   5   -1      0       -4       1   [[-1,0,1]]
    # dfs(nums, a, b+1, b+2, result)
    # 0   2   3   -1      1       2       2   [[-1,0,1]]
    # 0   2   4   -1      1       -1      -1  [[-1,0,1]]
    # 0   2   5   -1      1       -4      -4  [[-1,0,1]] 
    # 0   3   4   -1      2       -1      0   [[-1,0,1], [-1,2,-1]]
    # 0   3   5   -1      2       -4      -3   [[-1,0,1], [-1,2,-1]]
    # 0   4   5   -1      -1      -4      -6   [[-1,0,1], [-1,2,-1]]
    # dfs(nums, a+1, a+2, a+3, result)
    # 1   2   3   0       1       2       3   [[-1,0,1], [-1,2,-1]]
    # 1   2   4   0       1       -1      0    [[-1,0,1], [-1,2,-1], [1,-1,0]]
    # 1   2   5   0       1       -4      -3    [[-1,0,1], [-1,2,-1], [1,-1,0]]
    # dfs(nums, a, b+1, b+2, result)
    # 1   3   4   0       2       -1      1    [[-1,0,1], [-1,2,-1], [1,-1,0]]
    # 1   3   5   0       2       -4      -3    [[-1,0,1], [-1,2,-1], [1,-1,0]]
    # dfs(nums, a, b+1, b+2, result)
    # 1   4   5   0       -1      -4      -5      
    # dfs(nums, a+1, a+2, a+3, result)
    # 2   3   4   1       2       -1      2   
    # 2   3   5   1       2       -4      -1
    # dfs(nums, a, b+1, b+2, result)
    # 2   4   5   1       -1      -4      -4
    # dfs(nums, a+1, a+2, a+3, result)
    # 3   4   5   2       -1      -4      -3
    # result = [[-1,0,1], [-1,2,-1], [1,-1,0]]

# iterative, nested for loops, need to optimize
def threeSum(self, nums: List[int]) -> List[List[int]]:
    """given nums, return all unique triplets where a + b + c = 0"""
    result = []
    nums.sort()
    
    if len(nums) < 3:
        return result
    
    if len(nums) == 3 and (nums[0] + nums[1] + nums[2] == 0):
        return [nums]
    
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            for k in range(j+1,len(nums)):
                if nums[i]+nums[j]+nums[k] == 0:
                    if sorted([nums[i],nums[j],nums[k]]) not in result:
                        result.append(sorted([nums[i],nums[j],nums[k]]))
    
    return result

# using pointers and sliding window
def threeSum(nums):
    """given nums, return all unique triplets where a + b + c = 0
    >>> threeSum([-1,0,1,2,-1,-4])
    [[-1,-1,2],[-1,0,1]]
    >>> threeSum([0])
    []
    """

    result = []
    nums.sort()
    
    if len(nums) < 3:
        return result
    
    if len(nums) == 3 and (nums[0] + nums[1] + nums[2] == 0):
        return [nums]
    
    for i in range(len(nums)-2):
        j = i+1
        k = len(nums)-1
        
        while j < k:
            sum_ = nums[i] + nums[j] + nums[k]
            if sum_ == 0:
                if [nums[i],nums[j],nums[k]] not in result:
                    result.append([nums[i],nums[j],nums[k]])
                j += 1
                k -= 1
            elif sum_ < 0:
                j += 1 
            elif sum_ > 0:
                k -= 1
    
    return result

######### 101. Symmetric Tree ##############
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# recursion, 1st time
def isSymmetric(self, root: TreeNode) -> bool: 
    # binary tree: up to 2 children

    if not root:
        return True
    
    def dfs(n1, n2):
        if not n1 and not n2:
            return True
        if not n1 or not n2:
            return False
        if n1.val == n2.val:
            return dfs(n1.left, n2.right) and dfs(n1.right, n2.left)
        else:
            return False
        
        return True
    
    return dfs(root.left, root.right)


# recursion, 2nd time
def isSymmetric(self, root: TreeNode) -> bool:
    if not root:
        return True
    
    def dfs(n1, n2):
        if not n1 and not n2:
            return True
        if not n1 or not n2:
            return False
        if n1.val != n2.val:
            return False
        else:
            return dfs(n1.left, n2.right) and dfs(n1.right, n2.left)
        
        return True
    
    return dfs(root.left, root.right)

#                     1(n1)
#         n1.left         ==          n1.right
#         2 (n1)                       2 (n2)
# n1.left     n1.right           n2.left       n2.right= n1.right.right 
#     3 (n1)       4 (n2)         4 (n1)      3 (n2)

# iteration
def isSymmetric(self, root: TreeNode) -> bool:


######### 84. Largest Rectangle in Histogram ##############
def largestRectangleArea(heights):
    """
    Given n non-negative integers representing the histogram's bar height 
    where the width of each bar is 1, find the area of largest rectangle 
    in the histogram.
    >>> largestRectangleArea([2,1,5,6,2,3])
    10
    >>> largestRectangleArea([2,1,2])
    3
    """
    if not heights or max(heights)==0: 
        return 0
    
    if len(heights) == 1:
        return heights[0]
    
    stack = []
    area, i = 0, 0
    
    while i < len(heights):
        if not stack or heights[stack[-1]] <= heights[i]:
            stack.append(i)
            i += 1
        else:
            top = stack.pop()
            # shorter rectangle wiidth is i-stack[-1]; width for this rectangle is one smaller than that
            # if not stack, then this is the shortest rectangle, use i
            area = max(area, heights[top] * (i-stack[-1]-1 if stack else i))
            
    while stack:
        top = stack.pop()
        area = max(area, heights[top] * (i-stack[-1]-1 if stack else i))
        
    return area

#     [2,1,2]
#     i   stack   top stackafter  area
#     0   []      -   [0]         -
#     1   [0]     0   []          heights[0] * i = 2x1 = 2
#     1   []      -   [1]         -
#     2   [1]     -   [1,2]       -
#     3
#     3           2   [1]         heights[2] * i-stack[-1]-1 = 2x(3-1-1) = 2x1 = 2
#     3           1   []          heights[1] * i = 1x3 = 3


######### 226. Invert Binary Tree ##############
def invertTree(self, root: TreeNode) -> TreeNode:
    """invert binary tree"""
    
    if not root:
        return None
    
    temp = root.left
    root.left = root.right
    root.right = temp
    
    self.invertTree(root.left)
    self.invertTree(root.right)
    
    return root

# slightly more concise 
def invert_tree(node):
    """given binary tree, invert the tree"""
    # start at root
    if node:
        temp = node.left
        node.left = node.right
        node.right = temp
    
    dfs(node.left)
    dfs(node.right)
    
    return node

    

######### 3. Longest Substring Without Repeating Characters ##############
# using result string
def lengthOfLongestSubstring(self, s: str) -> int:
    """
    Given a string s, find the length of the longest substring without repeating characters.
    """       
    if not s:
        return 0
    
    res = ''
    length = 0

    for i in range(len(s)):
        if s[i] not in res:
            res += s[i]
        else:  
            # find length of current substring, then modify it
            length = max(length, len(res))
            first_idx = res.index(s[i])
            res = res[first_idx + 1:] + s[i]

    length = max(length, len(res))

    return length


# using sliding window, faster runtime: O(n)
def lengthOfLongestSubstring(self, s: str) -> int:
    """
    Given a string s, find the length of the longest substring without repeating characters.
    """  
    # initialize length at 0
    length, start, end = 0, 0, 0
    # create seen set, start/end pointers
    seen = set()
    # while loop (end < len(s)) and start <= end)
    while end < len(s) and start <= end:
        # if char not in seen, then add it and increment end pointer
        if s[end] not in seen:
            seen.add(s[end])
            end += 1
        # if char in seen, remove s[start] and add s[end]
        else:
            seen.remove(s[start])
            start += 1
        # length = max(length, end-start)
        length = max(length, end-start)
        # print(length)
    # return length
    return length
        

######### 70. Climbing Stairs ##############
# dp
def climbStairs(self, n: int) -> int:
    """given staircase of n steps, return int ways can climb to top"""
    # time and space complexity O(n): single loop, and dp is n long
    dp = [0] * (n+1)

    dp[0] = 1
    dp[1] = 1
    
    # can take one or two steps at a time so i-1 (1step) and i-2 (2steps)
    for i in range(2,len(dp)):
        dp[i] += dp[i-1] + dp[i-2]
    
    # print(dp)
    return dp[n]

# recursion without helper function
memo = {}

def climbStairs(n):
    """given staircase of n steps, return int ways can climb to top"""
    if n < 0: 
        return 0
    
    if n == 0 or n == 1:
        return 1
    
    if n not in self.memo:
        memo[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
    
    return memo[n]



# recursion with helper function
def climbStairs(self, n: int) -> int:
    """given staircase of n steps, return int ways can climb to top"""
    memo = {}
    
    def helper(n, memo):
        if n == 0:
            return 1
        if n == 1:
            return 1

        if n not in memo:
            memo[n] = helper(n-1, memo) + helper(n-2, memo)

        return memo[n] 

    return helper(n, memo)


######### 121. Best Time to Buy and Sell Stock ##############
def maxProfit(self, prices: List[int]) -> int:
    """return max profit or 0 if no profit"""

    # dp? maximum profit
    l = len(prices)
    dp = [0] * l # represent largest diff if buy on day i
    
    for i in range(1, l):
        for j in range(i+1, l):
            dp[i] = max(dp[i], prices[j]-prices[i])
    
    return max(dp)
    
    
    # concise one pass
    min_p = prices[0]
    max_p = 0
    
    for i in range(1, len(prices)):
        min_p = min(min_p, prices[i])
        max_p = max(max_p, prices[i]-min_p)
        
    return max_p

    # brute force, need to optimize
    diff = 0
    for i in range(len(prices)-1):
        for j in range(i+1, len(prices)):
            if prices[j]-prices[i] >=0:
                diff = max(diff, prices[j]-prices[i])
    
    return diff


######### 5. Longest Palindromic Substring ##############
def longestPalindrome(s):
    """given string s, return longest palindrome in s
    >>> longestPalindrome("babad")
    'bab'
    >>> longestPalindrome("cbbd")
    'bb'
    """
    # expand from middle
    def helper(s, l, r):
        # check to make sure l and r are in bounds and l-r contain a valid palindrome
        while l >= 0 and r < len(s) and s[l] == s[r]:
            # keep expanding l and r
            l -= 1
            r += 1
        # when loop exited, l and r are wider than the valid palindrome
        l += 1
        r -= 1
        # return length of longest palindrome and idx of start and finish (l and r)
        length = r-l+1
        return (length, l, r)
        
    length, l, r = 0, 0, 0
    # for loop to iterate through each i, both odd and even versions
    for i in range(len(s)):
        # odd, start at same letter
        odd_length, odd_l, odd_r = helper(s, i, i)
        if odd_length > length:
            length = odd_length
            l = odd_l
            r = odd_r
        
        # even, start at adjacent letters
        even_length, even_l, even_r = helper(s, i, i+1)
        if even_length > length:
            length = even_length
            l = even_l
            r = even_r
            
    return s[l:r+1]


######### 380. Insert Delete GetRandom O(1) ##############
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.RSet = set()

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.RSet:
            self.RSet.add(val)
            return True
        return False

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.RSet:
            self.RSet.remove(val)
            return True
        return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        from random import choice
        return choice(list(self.RSet))


######### 79. Word Search ##############
def exist(board, word):
    """Given an m x n board and a word, find if the word exists in the grid.
        The word can be constructed from letters of sequentially adjacent cells, 
        where "adjacent" cells are horizontally or vertically neighboring. 
        The same letter cell may not be used more than once.
        >>> exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED")
        True
        >>> exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE")
        True
        >>> exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB")
        False
    """
    
    def dfs(board, row, col, word, i): 
        if i == len(word):
            return True
        if 0 > row  or row >= len(board) or 0 > col or col >= len(board[0]):
            return False
        if board[row][col] != word[i]:
            return False

        # save cell value, process cell
        temp = board[row][col]
        board[row][col] = 0
            
        result = dfs(board, row-1, col, word, i+1)\
        or dfs(board, row+1, col, word, i+1)\
        or dfs(board, row, col-1, word, i+1)\
        or dfs(board, row, col+1, word, i+1)
    
        # restore cell value
        board[row][col] = temp
        
        return result
    
    # traverse board
    for row in range(len(board)):
        for col in range(len(board[0])):
            # look for first letter in word
            if board[row][col] == word[0] and dfs(board, row, col, word, 0): 
                return True
    
    return False


######### 872. Leaf-Similar Trees ##############
def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
    
    # take in root of tree, output leaf sequence from L to R
    def dfs(node, seq):
        if not node:
            return
        
        if not node.left and not node.right:
            seq.append(node.val)
        
        dfs(node.left, seq)
        dfs(node.right, seq)
        
        return seq

    # get leaf sequence for T1 and T2    
    seq_1 = dfs(root1, [])
    seq_2 = dfs(root2, [])

    return seq_1 == seq_2


######### 543. Diameter of Binary Tree ##############
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    """return length of diameter of tree"""

    # max diameter for each node is sum of L and R + node itself 
    # diameter d is an attribute of the class diameterOfBinaryTree
    self.d = 1
    def depth(node):
        if not node: return 0
        L = depth(node.left)
        R = depth(node.right)
        self.d = max(self.d, L+R+1)
        # print(self.d)
        return max(L, R) + 1

    depth(root)
    return self.d - 1


######### 111. Minimum Depth of Binary Tree ##############
# recursion using inner function (slowest)
def minDepth(self, root: TreeNode) -> int:
    """given binary tree, return min depth"""
    
    res = []
    
    if not root:
        return 0
    
    def dfs(node, depth):
        if node:
            if not node.right and not node.left:
                # print(depth)
                res.append(depth)
            return dfs(node.left, depth +1), dfs(node.right, depth+1)
    
    dfs(root, 1)
    
    return min(res)

# recursion, inner function, no appending (faster)
def minDepth(self, root: TreeNode) -> int:
    """given binary tree, return min depth"""
    
    if not root:
        return 0
    
    def dfs(node, depth):
        if not node:
            return depth
        if not node.right:
            return dfs(node.left, depth+1)
        if not node.left:
            return dfs(node.right, depth+1)
        else:
            return min(dfs(node.right, depth+1), dfs(node.left, depth+1))
        
    depth = dfs(root, 0)
    
    return depth


# recursion, no inner function (fastest)
def minDepth(self, root: TreeNode) -> int:
    """given binary tree, return min depth"""
    if not root:
        return 0
    if not root.right:
        return self.minDepth(root.left) +1
    if not root.left:
        return self.minDepth(root.right) +1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1


######### 525. Contiguous Array ##############
# brute force, appending, need to optimize
def findMaxLength(self, nums: List[int]) -> int:
    res = []
    if not nums:
        return len(res)
    count = 0
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            # print("subarray", nums[i:j+1])
            for digit in nums[i:j+1]:
                if digit == 0:
                    count += 1
                if digit == 1:
                    count -= 1
            if count == 0:
                res.append(len(nums[i:j+1]))
            count = 0
    if res: 
        return max(res)
    else:
        return 0

# brute force, using max, need to optimize
def findMaxLength(self, nums: List[int]) -> int:
    
    if not nums:
        return 0
    
    max_l, count = 0, 0
    
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            # print("subarray", nums[i:j+1])
            for digit in nums[i:j+1]:
                if digit == 0:
                    count += 1
                if digit == 1:
                    count -= 1
            if count == 0:
                max_l = max(max_l, len(nums[i:j+1]))
            count = 0
            
    return max_l

# using hm
def findMaxLength(self, nums: List[int]) -> int:
    max_l, count, length = 0, 0, 0
    # start it at -1 because index starts at 0
    d = {0:-1}
    for i in range(len(nums)):
        if nums[i]==0:
            count+=1
        if nums[i]==1:
            count-=1
        # print("i", i, "count", count)
        # if seen this count, then length = current pos - previous pos
        if count in d:
            length = i-d[count]
        if count not in d:
            d[count]=i            
        max_l = max(max_l, length)
        # print(d)
        
    return max_l


######### 404. Sum of Left Leaves ##############
# recursion, self attribute
def sumOfLeftLeaves(self, root: TreeNode) -> int:
    # traverse tree, dfs
    self.sum_ = 0
    
    def dfs(node, isLeft):
        # base case
        if not node:
            return

        # leaf node
        if not node.left and not node.right and isLeft:
            self.sum_ += node.val

        dfs(node.left, True)
        dfs(node.right, False)

    dfs(root, False)
    
    return self.sum_

# recursion, without additional attribute
def sumOfLeftLeaves(self, root: TreeNode) -> int:
    # traverse tree, dfs

    def dfs(node, isLeft):
        # base case
        if not node:
            return 0

        # leaf node
        if not node.left and not node.right and isLeft:
            return node.val

        return dfs(node.left, True) + dfs(node.right, False)

    ans = dfs(root, False)
    
    return ans


######### 993. Cousins in Binary Tree ##############
# doctests don't work because lack node class
def isCousins(root, x, y):
    """given tree, return True if nodes with x and y values are cousins
    >>> isCousins([1,2,3,null,4,null,5], 5, 4)
    True
    >>> isCousins([1,2,3,4], 3, 4)
    False
    """
    # cousins: same depth, diff parents
    # inner function return depth and parent as tuple
    res = []
    
    def dfs(node, parent, depth):
        if not node:
            return
        if node.val == x or node.val == y:
            res.append((depth, parent))
        dfs(node.left, node, depth+1)
        dfs(node.right, node, depth+1)
    
    dfs(root, None, 0)
    
    depth_x, parent_x = res[0]
    depth_y, parent_y = res[1]
    
    return depth_x == depth_y and parent_x != parent_y


######### 54. Spiral Matrix ##############
def spiralOrder(matrix) :
    """print matrix in spiral order
    >>> spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    [1,2,3,6,9,8,7,4,5]
    >>> spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    [1,2,3,4,8,12,11,10,9,5,6,7]
    """
    res = []
    if not matrix:
        return res
    
    m = len(matrix)
    n = len(matrix[0])
    left, right = 0, n - 1
    up, bottom = 0, m - 1
    
    direction = 0
    
    while left <= right and up <= bottom:
        if direction % 4 == 0:
            for i in range(left, right + 1):
                res.append(matrix[up][i])
            up += 1
        elif direction % 4 == 1:
            for i in range(up, bottom + 1):
                res.append(matrix[i][right])
            right -= 1
        elif direction % 4 == 2:
            for i in reversed(range(left, right + 1)):
                res.append(matrix[bottom][i])
            bottom -= 1
        else:
            for i in reversed(range(up, bottom + 1)):
                res.append(matrix[i][left])
            left += 1
        direction += 1
    
    return res
    

######### 74. Search a 2D Matrix ##############
# without binary search
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    for row in matrix:
        if len(row) == 0:
            continue
        # Checking wether the last element in the row is > or < target
        if row[-1] >= target:
            if target in row:
                return True
            return False


# binary search, slower
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    
    low, high = 0, m-1
    
    # searching for row
    while low <= high:
        mid = (low+high)//2
        if matrix[mid][0] == target:
            return True
        elif matrix[mid][0] < target:
            if matrix[mid][n-1] == target:
                return True
            if matrix[mid][n-1] > target:
                # check if target is in row
                return self.searchRow(matrix[mid], target, n)
            low = mid+1
        else:
            high = mid-1
        
    return False


# searching for target in row
def searchRow(self, row, target, l_row):
    low, high = 0, l_row-1
    
    while low <= high:
        mid = (low+high)//2
        if row[mid] > target:
            high = mid-1
        elif row[mid] < target:
            low = mid+1
        else:
            return True
    # if exit row and not found value, return False
    return False
        
        
######### 268. Missing Number ##############        
# set math
def missingNumber(self, nums: List[int]) -> int:
    # use set math
    range_set = set([x for x in range(len(nums)+1)])
    # print(range_set)
    ans = range_set.difference(nums)
    
    return ans.pop()


# using regular set, slower
def missingNumber(self, nums: List[int]) -> int:
    # use set
    num_set = set(nums)
    
    n = len(nums) + 1
    
    for num in range(n):
        if num not in num_set:
            return num


######### ##############
from collections import Counter
import heapq as hq

def topKFrequent(nums, k):
        """Given a non-empty array of integers, return the k most frequent elements
        >>> topKFrequent([1,1,1,2,2,3], 2)
        [1, 2]
        """
        
        if k == len(nums):
            return nums
        
        counter = Counter(nums)
        # print(counter)
       
        return hq.nlargest(k, counter.keys(), key=counter.get)


######### 236. Lowest Common Ancestor of a Binary Tree ##############
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    target = [p,q]
    def dfs(node, target):
        if not node:
            return
        # if node is either p or q, will return node
        if node in target:
            return node
        # otherwise, search to left and right of node
        left = dfs(node.left, target)
        right = dfs(node.right, target)
        # p and q are in left and right branches, LCA is the node
        if left and right:
            return node
        # if only L or R branch, will find first node that is p or q (child of itself)
        return left or right
    return dfs(root, target)

######### 110. Balanced Binary Tree ##############
# using height helper function, reattempt
def isBalanced(self, root: TreeNode) -> bool:
    """return True if balanced, False if not"""
    
    if not root: 
        return True
    
    # helper function to calculate height of a subtree from the root
    def tree_height(node):
        if not node:
            return 0
        left = tree_height(node.left)
        right = tree_height(node.right)
        
        return max(left, right) + 1
    
    L = tree_height(root.left)
    R = tree_height(root.right)
    
    if abs(L-R) > 1:
        return False
    
    return self.isBalanced(root.left) and self.isBalanced(root.right)


# using helper function to find height, not use boolean
def isBalanced(self, root: TreeNode) -> bool:
    """return True if balanced, False if not"""
    
    if not root:
        return True

    def dfs(node):
        """return height"""
        
        if not node:
            return 0
        print("node", node.val)
        left = dfs(node.left)
        print("L", left)
        right = dfs(node.right)
        print("R", right)
                
        return 1 + max(left, right)
    
    # check if root is balanced
    if abs(dfs(root.left) - dfs(root.right)) > 1:
        return False
    
    # check if left and right subtrees are balanced
    return self.isBalanced(root.left) and self.isBalanced(root.right)
    
# input
# [1,2,2,3,3,null,null,4,4]
# call stack order (left, left, left, hit end and then L and R, L and R)
# node 2
# node 3
# node 4
# L 0
# R 0
# L 1
# node 4
# L 0
# R 0
# R 1
# L 2
# node 3
# L 0
# R 0
# R 1
# node 2
# L 0
# R 0

# using helper function to find height and modify boolean; faster
def isBalanced(self, root: TreeNode) -> bool:
    """return True if balanced, False if not"""
    
    if not root:
        return True
    
    balanced = True

    def dfs(node):
        """return height, and also modified boolean"""
        if not node:
            return 0

        left = dfs(node.left)
        right = dfs(node.right)
        
        if abs(left - right) > 1:
            balanced = False
                
        return 1 + max(left, right)
    
    dfs(root)

    return balanced


# checking root and left/right subtrees, without boolean, slower (likely more function calls)
def isBalanced(self, root: TreeNode) -> bool:
    """return True if balanced, False if not"""
    
    if not root:
        return True

    def dfs(node):
        """return height, and also modified boolean"""
        if not node:
            return 0

        left = dfs(node.left)
        right = dfs(node.right)
                
        return 1 + max(left, right)
    
    # check if root is balanced
    if abs(dfs(root.left) - dfs(root.right)) > 1:
        return False
    # check if left and right subtrees are balanced
    return self.isBalanced(root.left) and self.isBalanced(root.right)
    
        

def isBalanced(self, root: TreeNode) -> bool:
    if not root:
        return True
    
    is_bal = True
    
    def find_height(node, height, is_bal):
        if not node: 
            return is_bal, height
        
        is_bal, left = find_height(node.left, height+1, is_bal)
        is_bal, right = find_height(node.right, height+1, is_bal)
        
        if is_bal:
            if abs(left-right) > 1:
                is_bal = False
        # return max height
        return is_bal, max(left, right)
    
    is_bal, height = find_height(root, 0, is_bal)
    
    return is_bal

# attempt 2
def isBalanced(self, root: TreeNode) -> bool:
    """return True if balanced, False if not"""
    
    # check height of L and R subtrees for each node: helper function to return height, 
    # stop cases: reach leaf node, max height of L and R are different by more than 1
    # can only return height in function, so use boolean to track if diff > 1
    
    if not root:
        return True
    
    diff = True
    
    def find_height(node, height, diff):
        if not node:
            # print("leaf", diff, height)
            return diff, height
        diff, L = find_height(node.left, height+1, diff)
        diff, R = find_height(node.right, height+1, diff)
        if abs(L-R) > 1:
            diff = False
        # print("max", max(L, R))
        return diff, max(L, R)

    diff, height = find_height(root, 0, diff)
    
    return diff
    
    


# [1,2,2,3,3,null,null,4,4]

# leaf True 4
# leaf True 4
# max 4
# leaf True 4
# leaf True 4
# max 4
# max 4
# leaf True 3
# leaf True 3
# max 3
# max 4
# leaf True 2
# leaf True 2
# max 2
# max 4

######### 208. Implement Trie (Prefix Tree) ##############
# using dictionary
# https://www.youtube.com/watch?v=hjUJFjcrbR4
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {'*':'*'}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        curr = self.root
        
        for char in word:
            # print("curr", curr, "char", char)
            if char not in curr:
                curr[char] = {}
            # move pointer down to that char's children
            curr = curr[char]
        # add in end symbol
        curr['*'] = {}
    
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        curr = self.root
        
        for char in word:
            if char not in curr:
                return False
            curr = curr[char]
        # if word has been inserted, there should be a *
        return "*" in curr

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        curr = self.root
        
        for char in prefix:
            if char not in curr:
                return False
            curr = curr[char]
        return True


######### 985. Sum of Even Numbers After Queries ##############
# brute force, need to optimize
def sumEvenAfterQueries(self, A: List[int], queries: List[List[int]]) -> List[int]:
    """given 2 lists, return list of int"""
    answer = [0]*len(queries)
    
    # loop through queries
    for i, item in enumerate(queries):
        A[item[1]] += item[0]
        even_sum = sum([el for el in A if el%2 == 0])
        answer[i] += even_sum
    
    return answer

# modifying sum
def sumEvenAfterQueries(self, A: List[int], queries: List[List[int]]) -> List[int]:
    """given 2 lists, return list of int"""
    answer = [0]*len(queries)
    prev_sum = sum([el for el in A if el%2 == 0])
    
    # loop through queries
    for i, item in enumerate(queries):
        # print("i", i, "prev sum", prev_sum)
        old_A = A[item[1]]   
        A[item[1]] += item[0]
        new_A = A[item[1]]
        # print("oldA", old_A, "new_A", new_A)
        # if A[i] was odd and now odd, then sum = prev
        # if A[i] was odd and now even, sum + newA[i]
        # if A[i] was even and now even, sum + item[0]
        # if A[i] was even and now odd, sum - oldA[i]
        
        if old_A %2 != 0:
            if new_A %2 == 0:
                sum_ = prev_sum + new_A
            else:
                sum_ = prev_sum
        if old_A %2 == 0:
            if new_A %2 == 0:
                sum_ = prev_sum + item[0]
            else:
                sum_ = prev_sum - old_A

        answer[i] += sum_
        prev_sum = sum_
        
    return answer

######### 657. Robot Return to Origin ##############
def judgeCircle(self, moves: str) -> bool:
    """given moves of robot, return True if robot returns to start"""
    
    return moves.count("U") == moves.count("D") and moves.count("L") == moves.count("R")


######### 961. N-Repeated Element in Size 2N Array ##############
# brute force, runtime O(N)
def repeatedNTimes(self, A: List[int]) -> int:
    # find len of list
    freq = len(A)/2
    # find element that has freq of len/2
    for el in A:
        if A.count(el) == freq:
            return el
    # return element

# not faster, uses Counter, runtime O(N)
def repeatedNTimes(self, A: List[int]) -> int:
    # make counter obj. return key of value of freq
    freq = len(A)/2
    
    a = Counter(A)

    for el, frequency in a.items():
        if frequency == freq:
            return el


######### 19. Remove Nth Node From End of List ##############
# second attempt
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    """given head node and int n apart, return list with n node removed"""

    # 2 pointers, one moves n+1 ahead
    # when faster one is none, the next node is the one to be removed
    # set curr.next to curr.next.next
    if not head.next:
        return
    
    fast = slow = head
    
    for i in range(n):
        fast = fast.next
    # if reached end of list and fast = None, supposed to remove head
    if not fast:
        return head.next

    while fast.next:
        fast = fast.next
        slow = slow.next
    
    slow.next = slow.next.next
    
    return head
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    """given head node and int n apart, return list with n node removed"""

    # 2 pointers, n+1 apart to have slow stop right before node to be removed
    if not head.next:
        return
    temp = ListNode(0)
    temp.next = head
    slow = fast = temp
    for i in range(n+1):
        fast = fast.next
    
    # advance both
    while fast:
        fast = fast.next
        slow = slow.next
    # when end reaches tail, slow should be right before node n
    # set slow.next to slow.next.next
    slow.next = slow.next.next
    # return head
    return temp.next
        
        
######### 217. Contains Duplicate ##############
def containsDuplicate(nums):
    """given list of int nums, return True if duplicates, False otherwise
    >>> containsDuplicate([1,2,3,1])
    True
    >>> containsDuplicate([1,2,3,4])
    False
    >>> containsDuplicate([1,1,1,3,3,4,3,2,4,2])
    True
    """
    # create set 
    # if len(set) == len(nums): return False
    # otherwise return True
    
    set_nums = set(nums)
    if len(nums) == len(set_nums):
        return False
    return True


######### 54. Spiral Matrix ##############
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        result = []
        
        while matrix:
            result += matrix.pop(0)
            # rotate matrix
            matrix = (list(zip(*matrix)))[::-1]
            print(matrix)
        
        return result
# for beginners who does not know the workings of zip here is explaination:

# l = [1,2,3]
# l2 = [4,5,6]

# print(list(zip(l,l2)))

# #it will print [(1,4),(2,5),(3,6)]

# For * (Star expression) = unpacking


# def add(a,b):
# 	return a+b
# l = (2,3)
# print(add(*l))
# It basically unpacks the tuple and puts them as positional arguments in the function call.

# 1 2 3
# 4 5 6
# 7 8 9

# [(4,7) (5,8) (6,9)]
# 69 58 47
# 54 87 = 87 54

######### 1290. Convert Binary Number in a Linked List to Integer ##############
def getDecimalValue(self, head: ListNode) -> int:
    # initialize binary nums array
    bin_nums = []
    sum_ = 0

    # while curr: start at head, append to binary nums
    curr = head
    while curr:
        bin_nums.append(curr.val)
        curr = curr.next
    # find len of binary nums array
    n = len(bin_nums)
    
    for i in range(n):
        sum_ += 2**i * bin_nums[n-1-i]
        
    return sum_


######### 142. Linked List Cycle II ##############
def detectCycle(self, head: ListNode) -> ListNode:
    if not head:
        return
    
    slow = fast = head
    
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        # found a cycle
        if slow == fast:
            break

    # no cycle present, reached end of list        
    if not fast or not fast.next:
        return
    
    # dist from where slow and fast meet to where cycle begins = dist from head to start of cycle
    while head != slow:
        head = head.next
        slow = slow.next
    return head


######### 160. Intersection of Two Linked Lists ##############
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    # traverse both lists
    # if 1=2, then return node
    A, B = headA, headB
    
    while A != B:
        A = headB if not A else A.next
        B = headA if not B else B.next
        # to keep the number of nodes traversed the same, keep switching between lists
    return A


######### 43. Multiply Strings ##############
def multiply(self, num1: str, num2: str) -> str:
    # edge cases
    if not num1 or not num2:
        return "0"
    
    if num1[0] == '0' or num2[0] == '0':
        return "0"
    
    # convert to int
    n1, n2 = 0, 0
    for digit in num1:
        n1 = n1 * 10 + ord(digit) - ord('0')
    for digit in num2:
        n2 = n2 * 10 + ord(digit) - ord('0')
    
    product = n1 * n2
    
    # convert to str
    res = ''
    while product:
        res += chr(ord('0') + product % 10)
        product //= 10
    
    return res[::-1]
    
    
######### 86. Partition List ##############
def partition(self, head: ListNode, x: int) -> ListNode:
    head_less = less = ListNode(None)
    head_greater = greater = ListNode(None)
    
    curr = head
    while curr:
        if curr.val < x:
            less.next = curr
            less = less.next
        else:
            greater.next = curr
            greater = greater.next
        curr = curr.next
    # important to prevent cycle
    greater.next = None
    less.next = head_greater.next
    
    return head_less.next


######### 2. Add Two Numbers ##############
# using modulo, faster
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    sum_ll_head = ListNode(None)
    curr = sum_ll_head
    arr = [0]

    while l1 or l2:
        new = ListNode()

        if l1:
            arr[-1] += l1.val
            l1 = l1.next
        if l2:
            arr[-1] += l2.val
            l2 = l2.next

        x = arr[-1] % 10
        new.val += x
        arr[-1] //= 10
        
        curr.next = new
        curr = curr.next
    
    if arr[-1] > 0:
        curr.next = ListNode(arr[-1])
        curr = curr.next
        
    return sum_ll_head.next

# using boolean, slower
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    sum_ll_head = ListNode(None)
    curr = sum_ll_head
    extra = False

    while l1 or l2:
        new = ListNode(0)

        if l1:
            new.val += l1.val
            l1 = l1.next
        if l2:
            new.val += l2.val
            l2 = l2.next

        if extra:
            new.val += 1
            extra = False
            
        if new.val > 9:
            extra = True
            new.val -= 10

        curr.next = new
        curr = curr.next
        
    if extra:
        curr.next = ListNode(1)
        
    return sum_ll_head.next

######### 232. Implement Queue using Stacks ##############

class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.s1.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            raise ValueError('empty q')
        
        # make s2 if empty, don't make again until need to replenish
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
                
        return self.s2.pop()

    
    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.s2:
            return self.s2[-1]
        else:
            return self.s1[0]


    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return not self.s1 and not self.s2

# using reverse method
class QueueWithStacks():
    def __init__(self):
        self.s1 = []
        self.s2 = []
    
    def empty(self):
        return len(self.s1) == 0 and len(self.s2) == 0
    
    def enqueue(self, item):
        self.s1.append(item)
    
    def reverse(self):
        while self.s1:
            pop = self.s1.pop()
            self.s2.append(pop)
    
    def dequeue(self):
        # if there's something in s1, make s2
        if len(self.s1) > 0 and not self.s2:
            self.reverse()
        return self.s2.pop()

    def peek(self):
        if len(self.s1) > 0 and not self.s2:
            self.reverse()
        return self.s2[-1]



######### 108. Convert Sorted Array to Binary Search Tree ##############
class TreeNode():
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# second attempt
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    if not nums:
        return
    
    median = len(nums) // 2
    
    root_val = nums[median]
    
    root = TreeNode(root_val)
    
    root.left = self.sortedArrayToBST(nums[:median])
    root.right = self.sortedArrayToBST(nums[median+1:])
    
    return root

# first attempt
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    if not nums:
        return
    
    m = len(nums)
    
    # find median of list
    idx_median = m //2
    
    root = TreeNode(nums[idx_median])
    
    root.left = self.sortedArrayToBST(nums[:idx_median])
    root.right = self.sortedArrayToBST(nums[idx_median+1:])
    
    return root


######### 102. Binary Tree Level Order Traversal ##############
# using regular q
def levelOrder(self, root: TreeNode) -> List[List[int]]:
    # bfs L before R
    if not root:
        return
    
    q = [root]
    
    res = []
    
    while q:
        level = []
        nodes = len(q)
        while nodes:
            pop = q.pop(0)
            nodes -= 1
            # print("pop", pop.val)
            if pop.left:
                q.append(pop.left)
            if pop.right:
                q.append(pop.right)
            level.append(pop.val)
        res.append(level)
    
    return res


# using collections deque, same speed
def levelOrder(self, root: TreeNode) -> List[List[int]]:
    if not root:
        return
    visited = set()
    q = collections.deque([root])
    ans = []
    
    visited.add(root)
    
    while q:
        level = []
        nodes = len(q)
        
        while nodes:
            pop = q.popleft()
            level.append(pop.val)
            nodes -= 1
            
            if pop.left and pop.left not in visited:
                visited.add(pop.left)
                q.append(pop.left)

            if pop.right and pop.right not in visited:
                visited.add(pop.right)
                q.append(pop.right)
                
        ans.append(level)
        
    return ans


######### 98. Validate Binary Search Tree ##############
# reattempt
def isValidBST(self, root: TreeNode) -> bool:
    if not root:
        print("end", root.val)
        return True
    
    # helper function to check BST rules of a node
    def BST(node, low, high):
        if not node:
            return True
        if low < node.val < high:
            return BST(node.left, low, node.val) and BST(node.right, node.val, high)
        return False
    
    return BST(root, float('-inf'), float('inf'))


# first attempt
def isValidBST(root):
    """return True if BST, False if not"""
    if not root:
        return True

    def dfs(node, lower, upper):
        """return True if follow BST rules"""
        if not node:
            return True
        # rules: L values < node.val, R values > node.val
        if lower < node.val < upper:
            return dfs(node.left, lower, node.val) and dfs(node.right, node.val, upper)
        return False
        

    return dfs(root, float('-inf'), float('inf'))

######### 572. Subtree of Another Tree ##############
def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    if not s:
        return False
    
    # helper function to call on roots
    def equal_trees(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        return equal_trees(node1.left, node2.left) and equal_trees(node1.right, node2.right)
    # need both conditions in case s == t, but not correct subtrees
    # otherwise will stop when s == t, but not right subtree
    if s.val == t.val and equal_trees(s,t):
        return equal_trees(s,t)  # or return True
    
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
    if not s:
        return False
    
    # helper function to call on root nodes of 2 subtrees
    def equal(s, t):
        if not s and not t:
            return True
        if not s or not t:
            return False
        if s.val != t.val: 
            return False
        return equal(s.left, t.left) and equal(s.right, t.right)
    
    if s.val == t.val and equal(s,t):
        return True
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

######### 894. All Possible Full Binary Trees ##############
def allPossibleFBT(self, n: int) -> List[TreeNode]:
    
    # even number of nodes: not full tree
    if n % 2 == 0:
        return []
    
    memo = {}
    
    def make_full_bt(n):
        # base cases
        if n == 1:
            return [TreeNode(0)]
        if n in memo:
            return memo[n]
        
        # if n not in memo, make entry for n
        res = []
        
        for i in range(1, n, 2):
            # make all possible combos of total num of L and R nodes 
            left = make_full_bt(i)
            right = make_full_bt(n-i-1)
            # add subtree to res
            res.extend([TreeNode(0, l, r) for l in left for r in right])
            
        memo[n] = res
                
        return memo[n]
    
    return make_full_bt(n)


######### 606. Construct String from Binary Tree ##############
# no helper function
def tree2str(self, t: TreeNode) -> str:
    # output preorder nodes
    # no left child and there is a right child, we need to print () for the left child
    # when we print a child, print ( child )
    
    if not t:
        return ""
    
    # if no L child
    if not t.left:
        # if also no R child, just return str(t.val)
        if not t.right:
            return str(t.val) 
        # if R child, then have to put in the () for the null L child
        else:
            return str(t.val) + "()(" + self.tree2str(t.right) + ")"
    # if no R child, just print out root with the L child
    if not t.right:
        return str(t.val) + "(" + self.tree2str(t.left) + ")"
    # if both L and R children, then print both out
    return str(t.val) + "(" + self.tree2str(t.left) + ")(" + self.tree2str(t.right) + ")"   


######### 112. Path Sum ##############
def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
    if not root:
        return False
    
    # if leaf node and targetSum - root.val == 0: return True
    if not targetSum-root.val and not (root.left or root.right):
        return True
    # neither of those conditions is true, check children
    # return whichever one is True, if both are not True, will return False
    return self.hasPathSum(root.left, targetSum-root.val) or self.hasPathSum(root.right, targetSum-root.val)


######### 63. Unique Paths II ##############
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    memo = {}
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    
    def ways(r, c, memo):
        # out of bounds or an obstacle, no way to get there
        if r < 0 or c < 0 or obstacleGrid[r][c] == 1:
            return 0
        # only one way to get to origin from origin
        if r == c == 0:
            return 1
        if r >= m or c >= n:
            return 0
        if (r,c) not in memo:
            memo[(r,c)] = ways(r-1,c, memo) + ways(r, c-1, memo)
        
        return memo[(r,c)]
    
    return ways(m-1, n-1, memo)


# another recursion, opposite base case
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    cache = {}
    max_row = len(obstacleGrid) - 1 
    max_column = len(obstacleGrid[0]) - 1 
    
    def helper(obstacleGrid, current_row, current_column, cache): 
        if current_row > max_row or current_column > max_column:
            return 0 

        if obstacleGrid[current_row][current_column] == 1:
            return 0 

        if current_row == max_row and current_column == max_column:
            return 1 

        if (current_row, current_column) not in cache:
            cache[(current_row, current_column)] = helper(obstacleGrid, current_row + 1, current_column, cache) + helper(obstacleGrid, current_row, current_column + 1, cache) 

        return cache[(current_row, current_column)]

    return helper(obstacleGrid, 0, 0, {})


######### 46. Permutations ##############
def permute(self, nums: List[int]) -> List[List[int]]:
    if not nums:
        return []
    if len(nums) == 1:
        return [nums]
    l = []
    for i in range(len(nums)):
        new_nums = nums[:i] + nums[i+1:]
        n = nums[i]
        for p in self.permute(new_nums):
            l.append([n] + p)     
    return l    


######### 1791. Find Center of Star Graph ##############
from collections import Counter

def findCenter(self, edges: List[List[int]]) -> int:
    # undirected: equivalent either direction
    d = {}
    for edge in edges:
        if edge[1] in d:
            return edge[1]
        if edge[1] in d.values():
            return edge[1]
        d[edge[0]] = edge[1]


######### Permutations II ##############
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    if not nums:
        return []
    if len(nums) == 1:
        return [nums]
    res = []
    for i in range(len(nums)):
        head = [nums[i]]
        new_nums = nums[:i] + nums[i+1:]
        for seq in self.permuteUnique(new_nums):
            if head + seq not in res:
                res.append(head + seq)
    return res


######### 128. Longest Consecutive Sequence ##############
# using set, runtime O(n)
def longestConsecutive(self, nums: List[int]) -> int:
    """given int array, return len of longest consecutive seq"""
    if not nums:
        return 0
    
    nums_set = set(nums)
    longest = 1
    
    for num in nums_set:
        # starting out with the smallest num not in seq
        if num - 1 not in nums_set:
            curr = num
            # print(curr)
            curr_longest = 1
            while curr + 1 in nums_set:
                curr += 1
                curr_longest += 1
                # print("curr longest", curr_longest)
            longest = max(longest, curr_longest)
        return longest

# using sort, runtime O(n log n)
def longestConsecutive(self, nums: List[int]) -> int:
    """given int array, return len of longest consecutive seq"""
    if not nums:
        return 0
    
    nums.sort()
    
    curr, longest = 1,1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            if nums[i] == nums[i-1] + 1:
                curr += 1
            else:
                longest = max(curr, longest)
                curr = 1

    return max(longest, curr)

# using counter to reduce duplicates, and also dp, very slow, runtime O(n2)
from collections import Counter

def longestConsecutive(self, nums: List[int]) -> int:
    """given int array, return len of longest consecutive seq"""
    if not nums:
        return 0
    
    c = Counter(nums)
    l_nums = list(c.keys())
    # sort the keys of c to get max length possible
    l_nums.sort()
    print(l_nums)

    dp = [1] * len(l_nums)
    
    for i in range(len(l_nums)):
        for j in range(i):
            if l_nums[i] == l_nums[j] + 1:
                dp[i] = max(dp[i], dp[j] +1)
    
    return max(dp)


#########  ##############
#########  ##############
#########  ##############
#########  ##############
#########  ##############
#########  ##############
#########  ##############
#########  ##############

# if __name__ == '__main__':
import doctestxs

print()
result = doctest.testmod()
if not result.failed:
    print("ALL TESTS PASSED!")
print()