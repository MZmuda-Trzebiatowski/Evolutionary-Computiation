#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <numeric>
#include <random>

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
            d[i][j] = round(eu);
        }
    }
    return d;
}


// Compute the objective value of a tour
long long tour_objective(const vector<int> &tour, const vector<vector<int>> &d, const vector<Node> &nodes) {
    long long sum_d = 0;
    int m = tour.size();
    for(int i=0;i<m;i++){
        int a = tour[i];
        int b = tour[(i+1)%m]; // wrap around
        sum_d += d[a][b];
    }
    long long sum_cost = 0;
    for(int v : tour) sum_cost += nodes[v].cost;
    return sum_d + sum_cost;
}


// Random selection of k nodes from n
vector<int> random_solution(int n, int k) {
    vector<int> ids(n);
    iota(ids.begin(), ids.end(), 0);
    shuffle(ids.begin(), ids.end(), mt19937{random_device{}()});
    ids.resize(k);
    return ids;
}


// NN with end-only insertion
vector<int> nn_end(int start_node, int k, const vector<vector<int>>& d, const vector<Node>& nodes) {
    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.push_back(start_node);
    used[start_node] = true;
    while((int)cycle.size() < k) {
        int best_node = -1;
        long long best_delta = LLONG_MAX/4;
        int iidx = cycle.back();
        int jidx = cycle.front();
        int base_edge = d[iidx][jidx];
        for(int cand=0;cand<n;cand++){
            if(used[cand]) continue;
            long long delta = (long long)d[iidx][cand] + d[cand][jidx] - base_edge + nodes[cand].cost;
            if(delta < best_delta) { best_delta = delta; best_node = cand; }
        }
        if(best_node==-1) break; 
        cycle.push_back(best_node);
        used[best_node]=true;
    }
    return cycle;
}


// NN with any-position insertion
vector<int> nn_anypos(int start_node, int k, const vector<vector<int>>& d, const vector<Node>& nodes) {
    int n = d.size();
    vector<char> used(n,false);
    vector<int> cycle;
    cycle.push_back(start_node);
    used[start_node]=true;
    while((int)cycle.size() < k) {
        int best_node = -1;
        int best_pos = -1; 
        long long best_delta = LLONG_MAX/4;
        int m = cycle.size();
        for(int cand=0;cand<n;cand++){
            if(used[cand]) continue;
            for(int pos=0; pos<m; pos++){
                int a = cycle[pos];
                int b = cycle[(pos+1)%m];
                long long delta = (long long)d[a][cand] + d[cand][b] - d[a][b] + nodes[cand].cost;
                if(delta < best_delta){
                    best_delta = delta;
                    best_node = cand;
                    best_pos = pos;
                }
            }
        }
        if(best_node==-1) break;
        cycle.insert(cycle.begin() + best_pos + 1, best_node);
        used[best_node]=true;
    }
    return cycle;
}


void export_tour_svg(const string& filename, const vector<int>& tour, const vector<Node>& nodes) {
    if (tour.empty()) {
        cerr << "Warning: Cannot export empty tour.\n";
        return;
    }

    // 1. Determine bounding box for scaling
    int min_x = nodes[0].x, max_x = nodes[0].x;
    int min_y = nodes[0].y, max_y = nodes[0].y;
    int min_cost = nodes[0].cost, max_cost = nodes[0].cost;

    for (const auto& node : nodes) {
        min_x = min(min_x, node.x);
        max_x = max(max_x, node.x);
        min_y = min(min_y, node.y);
        max_y = max(max_y, node.y);
        min_cost = min(min_cost, node.cost);
        max_cost = max(max_cost, node.cost);
    }

    // 2. Define SVG canvas parameters
    const int SVG_WIDTH = 1920;
    const int SVG_HEIGHT = 1080;
    const int PADDING = 40; // Space from the edge

    double scale_x = (double)(SVG_WIDTH - 2 * PADDING) / (max_x - min_x + 1);
    double scale_y = (double)(SVG_HEIGHT - 2 * PADDING) / (max_y - min_y + 1);
    double scale = min(scale_x, scale_y); // Use the smaller scale factor to maintain aspect ratio

    // Function to scale coordinates
    auto scale_coord_x = [&](int x) {
        return PADDING + (x - min_x) * scale;
    };
    auto scale_coord_y = [&](int y) {
        // SVG y-axis is top-down, so we invert the scaling
        return SVG_HEIGHT - PADDING - (y - min_y) * scale; 
    };
    
    // Function to scale cost to radius (min radius 3, max radius 15)
    auto scale_cost_to_radius = [&](int cost) {
        if (max_cost == min_cost) return 6.0;
        double normalized = (double)(cost - min_cost) / (max_cost - min_cost);
        return 3.0 + normalized * 12.0; // Radius between 3 and 15
    };


    // 3. Open file and write SVG header
    ofstream out(filename);
    if (!out) {
        cerr << "Error: cannot create SVG file " << filename << "\n";
        return;
    }

    out << "<svg width=\"" << SVG_WIDTH << "\" height=\"" << SVG_HEIGHT << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    // Background color
    out << "  <rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n";
    // Text label for the best objective value
    auto d_matrix = compute_distance_matrix(nodes); // Recompute or pass d
    long long obj = tour_objective(tour, d_matrix, nodes); 
    out << "  <text x=\"" << PADDING << "\" y=\"" << PADDING / 2.0 << "\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#333\">\n";
    out << "    Best Objective: " << obj << " | Tour Size: " << tour.size() << "\n";
    out << "  </text>\n";


    // 4. Draw edges (lines)
    out << "  \n";
    int m = tour.size();
    for (int i = 0; i < m; ++i) {
        int idx1 = tour[i];
        int idx2 = tour[(i + 1) % m];

        double x1 = scale_coord_x(nodes[idx1].x);
        double y1 = scale_coord_y(nodes[idx1].y);
        double x2 = scale_coord_x(nodes[idx2].x);
        double y2 = scale_coord_y(nodes[idx2].y);

        out << "  <line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\"\n";
        out << "        stroke=\"#0000FF\" stroke-width=\"2\" stroke-dasharray=\"4,2\"/>\n";
    }

    // 5. Draw nodes (circles)
    out << "  \n";
    int start_node_idx = tour[0]; // Get the index of the first node in the tour

    for (size_t i = 0; i < nodes.size(); ++i) {
        int current_x = nodes[i].x;
        int current_y = nodes[i].y;
        int current_cost = nodes[i].cost;
        
        double cx = scale_coord_x(current_x);
        double cy = scale_coord_y(current_y);
        double r = scale_cost_to_radius(current_cost);
        
        // Check if the node is in the tour
        bool in_tour = (find(tour.begin(), tour.end(), i) != tour.end());

        string fill_color;
        string stroke_color;
        double stroke_width = 1.5; // Default stroke width

        if (i == start_node_idx) { // This is the starting node
            fill_color = "#00AA00"; // Green color
            stroke_color = "#006400"; // Darker green border
            stroke_width = 3.0; // Thicker border for start node
        } else if (in_tour) { // Other selected nodes
            fill_color = "#FF0000"; // Red color
            stroke_color = "#8B0000"; // Darker red border
        } else { // Unselected nodes
            fill_color = "#AAAAAA"; // Grey color
            stroke_color = "#666666"; // Darker grey border
        }

        out << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r << "\"\n";
        out << "          fill=\"" << fill_color << "\" stroke=\"" << stroke_color << "\" stroke-width=\"" << stroke_width << "\">\n";
        out << "    <title>Node " << i << " (x:" << current_x << ", y:" << current_y << ", cost:" << current_cost << (i == start_node_idx ? ", START" : "") << ")</title>\n";
        out << "  </circle>\n";
    }

    // 6. Close SVG tag
    out << "</svg>\n";
    out.close();

    // cout << "SVG visualization saved to: " << filename << "\n";
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

    const int N_SOL = 200;

    long long best_random_obj = LLONG_MAX/4;
    vector<int> best_random_tour;


    // Generate N_SOL random solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        auto tour = random_solution(n, k);
        long long obj = tour_objective(tour, d, nodes);
        if(obj < best_random_obj){
            best_random_obj = obj;
            best_random_tour = tour;
        }
    }
    cout<<"Random: best objective = "<<best_random_obj<<"\n";
    export_tour_svg("best_random_tour.svg", best_random_tour, nodes);


    long long best_nn_end_obj = LLONG_MAX/4;
    vector<int> best_nn_end_tour;

    // Generate N_SOL nn_end solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = nn_end(start, k, d, nodes);
        long long obj = tour_objective(tour, d, nodes);
        if(obj < best_nn_end_obj){
            best_nn_end_obj = obj;
            best_nn_end_tour = tour;
        }
    }
    cout<<"NN_end: best objective = "<<best_nn_end_obj<<"\n";
    export_tour_svg("best_end_nn_tour.svg", best_nn_end_tour, nodes);


    long long best_nn_any_obj = LLONG_MAX/4;
    vector<int> best_nn_any_tour;

    // Generate N_SOL NN any-position and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = nn_anypos(start, k, d, nodes);
        long long obj = tour_objective(tour, d, nodes);
        if(obj < best_nn_any_obj){
            best_nn_any_obj = obj;
            best_nn_any_tour = tour;
        }
    }
    cout<<"NN_any_position: best objective = "<<best_nn_any_obj<<"\n";
    export_tour_svg("best_anypos_nn_tour.svg", best_nn_any_tour, nodes);

    return 0;
}
