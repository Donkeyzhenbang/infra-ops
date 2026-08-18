#include <iostream>
#include <vector>

using namespace std;
//! 递归题目 函数定义 只想当前层 剩下交给递归 找终止条件
struct ListNode {
    int val;
    ListNode* next;
};

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */ 
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(list1 == nullptr) return list2;
        if(list2 == nullptr) return list1;
        if(list1->val < list2->val){
            list1->next = mergeTwoLists(list1->next, list2);
            return list1;
        }else{
            list2->next = mergeTwoLists(list1, list2->next);
            return list2;
        }
    }
};

int main()
{
    // 构造 list1: 1 -> 2 -> 4
    ListNode* list1 = new ListNode{1, new ListNode{2, new ListNode{4, nullptr}}};

    // 构造 list2: 1 -> 3 -> 4
    ListNode* list2 = new ListNode{1, new ListNode{3, new ListNode{4, nullptr}}};

    Solution s;
    ListNode* merged = s.mergeTwoLists(list1, list2);

    // 输出合并结果
    for (ListNode* p = merged; p != nullptr; p = p->next) {
        cout << p->val << " ";
    }
    cout << endl;

    return 0;
}