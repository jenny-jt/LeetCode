
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
class Solution:
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

