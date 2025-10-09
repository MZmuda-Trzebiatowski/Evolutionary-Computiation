#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>


using namespace std;

struct Node {
    int x, y, cost;
};


vector<Node> read_instance(const string &fname) {
    ifstream in(fname);
    if(!in) {
        cerr << "Error: cannot open file " << fname << "\n";
        exit(1);
    }

    vector<Node> nodes;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;

        for (char &c : line) {
            if (c == ';') c = ' ';
        }

        stringstream ss(line);
        int x, y, cost;
        if (!(ss >> x >> y >> cost)) {
            cerr << "Warning: could not parse line -> " << line << "\n";
            continue;
        }
        nodes.push_back({x, y, cost});
    }

    return nodes;
}


vector<vector<int>> compute_distance_matrix(const vector<Node> &nodes) {
    int n = nodes.size();
    vector<vector<int>> d(n, vector<int>(n, 0));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j) { d[i][j]=0; continue; }
            double dx = nodes[i].x - nodes[j].x;
            double dy = nodes[i].y - nodes[j].y;
            double eu = sqrt(dx*dx + dy*dy);
            d[i][j] = (int)lround(eu);
        }
    }
    return d;
}


int main(int argc, char** argv) {
    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " <instance-file>\n";
        return 1;
    }
    string fname = argv[1];
    vector<Node> nodes = read_instance(fname);
    int n = nodes.size();
    if(n==0) { cerr<<"No nodes read from file.\n"; return 1; }
    
    int k = (n + 1) / 2; // half of the nodes, rounded up
    cout<<"Read "<<n<<" nodes; selecting k = "<<k<<" nodes per solution.\n";

    auto d = compute_distance_matrix(nodes);

    return 0;
}
