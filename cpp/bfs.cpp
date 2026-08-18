#include <iostream>
#include <vector>
#include <queue>
using namespace std;

int main() {
    // 1. 建图 (0:A, 1:B, 2:C, 3:D)
    int n = 4;
    vector<vector<int>> adj(n); // 邻接表
    vector<int> indegree(n, 0); // 入度数组

    // A(0) -> C(2)
    adj[0].push_back(2);
    indegree[2]++;
    // B(1) -> C(2)
    adj[1].push_back(2);
    indegree[2]++;
    // C(2) -> D(3)
    adj[2].push_back(3);
    indegree[3]++;

    // 2. 初始化队列，入度为0的节点入队（第一层）
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) q.push(i);
    }

    // 3. 按层遍历（核心：记录每层size）
    vector<char> name = {'A', 'B', 'C', 'D'};
    int level = 1;

    while (!q.empty()) {
        // 关键点：固定住当前层的节点个数
        int level_size = q.size(); 
        cout << "第 " << level << " 层: ";

        // 只弹出当前层的节点
        for (int i = 0; i < level_size; i++) {
            int node = q.front();
            q.pop();
            cout << name[node] << " ";

            // 遍历邻居，入度减1，如果变为0则入队（属于下一层）
            for (int neighbor : adj[node]) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
        level++;
    }

    return 0;
} 