#include <iostream>
#include <vector>

using namespace std;

void quick_sort(vector<int>& nums, int left, int right)
{
    if(left >= right) return;
    int pivot = nums[left + (right - left) / 2];
    int i = left;
    int j = right;
    while(i <= j){
        while(nums[i] < pivot) i ++;
        while(nums[j] > pivot) j --;
        if(i <= j) {
            swap(nums[i], nums[j]);
            i ++;
            j --;
        }
    }
    quick_sort(nums, left, j);
    quick_sort(nums,i, right);
}

int main()
{
    vector<int> a = {2, 4, 6, 3, 7 ,9, 1};
    quick_sort(a, 0, a.size() - 1);
    for(int x : a)  
        cout << x << " ";

}