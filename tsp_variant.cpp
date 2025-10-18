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


long long get_increase(const vector<Node> &nodes, const vector<vector<int>> &d, int node1_id, int node2_id) {
    long long distance_and_cost = 0;
    distance_and_cost += d[node1_id][node2_id];
    distance_and_cost += nodes[node2_id].cost;

    return distance_and_cost;
}


vector<int> nn_end(int n, const vector<Node> &nodes, const vector<vector<int>> &d, int starting_node_id) {
    int filled_nodes = 0;
    vector<int> tour(n, -1);

    std::vector<int> remining_nodes(200);
    for (int i = 0; i < 200; ++i) {
        remining_nodes[i] = i;
    }

    int end_node_id = starting_node_id;
    tour[filled_nodes] = starting_node_id;
    filled_nodes += 1;
    remining_nodes.erase(remove(remining_nodes.begin(), remining_nodes.end(), starting_node_id), remining_nodes.end());
    

    for(; filled_nodes < n; filled_nodes++) {
        int best_next = remining_nodes.front();
        long long min_next = get_increase(nodes, d, end_node_id, best_next);
        for(int next_node: remining_nodes) {
            long long next_node_increase = get_increase(nodes, d, end_node_id, next_node);
            if (next_node_increase < min_next)
            {
                min_next = next_node_increase;
                best_next = next_node;
            }
            
        }
        tour[filled_nodes] = best_next;
        end_node_id = best_next;
        remining_nodes.erase(remove(remining_nodes.begin(), remining_nodes.end(), best_next), remining_nodes.end());
    }

    return tour;
}


long long get_increase_anypos(const vector<Node> &nodes, const vector<vector<int>> &d,
                              int a, int b, int new_node_id) {
    long long delta = 0;
    delta += d[a][new_node_id] + d[new_node_id][b] - d[a][b];
    delta += nodes[new_node_id].cost;
    return delta;
}


vector<int> nn_any_position(int n, const vector<Node> &nodes, const vector<vector<int>> &d,
                            int starting_node_id) {
    int filled_nodes = 0;
    vector<int> tour;
    tour.reserve(n);

    vector<int> remaining_nodes(n);
    iota(remaining_nodes.begin(), remaining_nodes.end(), 0);

    tour.push_back(starting_node_id);
    filled_nodes = 1;
    remaining_nodes.erase(remove(remaining_nodes.begin(), remaining_nodes.end(), starting_node_id),
                          remaining_nodes.end());


    if (!remaining_nodes.empty()) {
        int best_next = remaining_nodes.front();
        long long min_next = get_increase(nodes, d, starting_node_id, best_next);
        for (int next_node : remaining_nodes) {
            long long next_node_increase = get_increase(nodes, d, starting_node_id, next_node);
            if (next_node_increase < min_next) {
                min_next = next_node_increase;
                best_next = next_node;
            }
        }
        tour.push_back(best_next);
        filled_nodes++;
        remaining_nodes.erase(remove(remaining_nodes.begin(), remaining_nodes.end(), best_next),
                              remaining_nodes.end());
    }

    while ((int)tour.size() < n && !remaining_nodes.empty()) {
        long long best_increase = LLONG_MAX;
        int best_node = -1;
        int best_pos = -1;

        int m = tour.size();
        for (int candidate : remaining_nodes) {
            for (int i = 0; i < m; ++i) {
                int a = tour[i];
                int b = tour[(i + 1) % m];  // wrap to make it a cycle
                long long increase = get_increase_anypos(nodes, d, a, b, candidate);
                if (increase < best_increase) {
                    best_increase = increase;
                    best_node = candidate;
                    best_pos = i + 1; // insert after position i
                }
            }
        }

        if (best_node != -1) {
            tour.insert(tour.begin() + best_pos, best_node);
            remaining_nodes.erase(remove(remaining_nodes.begin(), remaining_nodes.end(), best_node),
                                  remaining_nodes.end());
        } else {
            break; 
        }
    }

    return tour;
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

    cout << "SVG visualization saved to: " << filename << "\n";
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
        auto tour = nn_end(k, nodes, d, t);
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
        auto tour = nn_any_position(k, nodes, d, start);
        long long obj = tour_objective(tour, d, nodes);
        if(obj < best_nn_any_obj){
            best_nn_any_obj = obj;
            best_nn_any_tour = tour;
        }
    }
    cout<<" NN_any_position: best objective = "<<best_nn_any_obj<<"\n";
    export_tour_svg("best_anypos_nn_tour.svg", best_nn_any_tour, nodes);

    return 0;
}
