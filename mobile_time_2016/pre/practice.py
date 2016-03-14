class solution:
	def search(self, nums, target):
		"""
		:type nums: List[int]
		:type target: int
		:rtype: int
		"""
		first = 0
		last = len(nums)
		while(first!=last):
			mid = (first+last)/2
			if(nums[mid] == target):
				return mid
			if(nums[mid]>=nums[first]):
				if((target>=nums[first]) & (target<nums[mid])):
					last = mid
				else:
					first = mid+1
			else:
				if((target>nums[mid]) & (target<=nums[last-1])):
					print nums[last-1]
					first = mid+1
				else:
					last = mid
		return -1
sol = solution()
nums = [3,1]
a = sol.search(nums,3)
print a