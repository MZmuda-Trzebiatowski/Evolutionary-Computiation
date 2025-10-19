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
    cycle.reserve(k);
    cycle.push_back(start_node);
    used[start_node] = true;
    while((int)cycle.size() < k) {
        int best_node = -1;
        long long best_delta = LLONG_MAX;
        int iidx = cycle.back();

        for(int cand=0; cand<n; cand++){
            if(used[cand]) continue;
            long long delta = (long long)d[iidx][cand] + nodes[cand].cost;
            if(delta < best_delta) { best_delta = delta; best_node = cand; }
        }
        if(best_node==-1) break; 
        cycle.push_back(best_node);
        used[best_node]=true;
    }
    return cycle;
}


vector<int> nn_anypos(int start_node, int k, const vector<vector<int>>& d, const vector<Node>& nodes) {
    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.reserve(k);

    cycle.push_back(start_node);
    used[start_node] = true;


    while ((int)cycle.size() < k)
    {
        int best_put_after_index = -1;
        int next_node = -1;
        long long best_at_start_val = LLONG_MAX;

        // check for inserting at start
        for (int cand = 0; cand < n; cand++)
            {
                if (used[cand]) continue;
                long long val = (long long)d[cand][cycle[0]] + nodes[cand].cost;               

                if (val < best_at_start_val)
                {
                    best_at_start_val = val;
                    next_node = cand;
                }
            }

        long long best_nn = best_at_start_val;
        

        int c_size = (int)cycle.size();
        for (int i = 0; i < c_size; i++)
        {
            int best_next_node_locally = -1;
            long long bnnl_min = LLONG_MAX;
            int put_after = cycle[i];
            int put_before = cycle[(i + 1) % c_size];

            for (int cand = 0; cand < n; cand++)
            {
                if (used[cand]) continue;
                long long val = (long long)d[put_after][cand] + nodes[cand].cost + d[cand][put_before];
                if (i < ((i + 1) % c_size))
                {
                    val -= d[put_after][put_before];
                }
                

                if (val < bnnl_min)
                {
                    bnnl_min = val;
                    best_next_node_locally = cand;
                }
            }

            if (bnnl_min < best_nn)
            {
                best_nn = bnnl_min;
                next_node = best_next_node_locally;
                best_put_after_index = i;
            }
        }
        
        cycle.insert(cycle.begin() + best_put_after_index + 1, next_node);
        used[next_node] = true;
    }

    return cycle;
}



// Greedy cycle construction
vector<int> greedy_cycle(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.reserve(k);

    cycle.push_back(start_node);
    used[start_node] = true;

    int best_second = -1;
    long long best_delta = LLONG_MAX;

    // add second node to form first edge
    for (int cand = 0; cand < n; cand++)
    {
        if (used[cand])
            continue;
        long long delta = (long long)d[start_node][cand] + nodes[cand].cost;
        if (delta < best_delta)
        {
            best_delta = delta;
            best_second = cand;
        }
    }

    cycle.push_back(best_second);
    used[best_second] = true;

    while ((int)cycle.size() < k)
    {
        int best_put_after_index = 0;
        int next_node = -1;
        long long best_nn = LLONG_MAX;
        int c_size = (int)cycle.size();
        for (int i = 0; i < c_size; i++)
        {
            int best_next_node_locally = -1;
            long long  bnnl_min = LLONG_MAX;
            int put_after = cycle[i];
            int put_before = cycle[(i + 1) % c_size];

            for (int cand = 0; cand < n; cand++)
            {
                if (used[cand]) continue;
                long long val = (long long)d[put_after][cand] + nodes[cand].cost + d[cand][put_before];      
                if (c_size > 2)
                {
                    val -= d[put_after][put_before];
                }       

                if (val < bnnl_min)
                {
                    bnnl_min = val;
                    best_next_node_locally = cand;
                }
            }

            if (bnnl_min < best_nn)
            {
                best_nn = bnnl_min;
                next_node = best_next_node_locally;
                best_put_after_index = i;
            }
        }

        cycle.insert(cycle.begin() + best_put_after_index + 1, next_node);
        used[next_node] = true;
    }

    return cycle;
}

vector<int> nn_anypos_regret(int start_node, int k, const vector<vector<int>>& d, const vector<Node>& nodes, double w1, double w2) {
    struct RegretRankingEntry {
            int node;
            int first_pos;
            long long first_val;
            int second_pos;
            long long second_val;
        };

    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.reserve(k);

    cycle.push_back(start_node);
    used[start_node] = true;


    while ((int)cycle.size() < k)
    {
        
        vector<RegretRankingEntry> ranking;

        for (int i = 0; i < n; i++)
        {
            if (used[i]) continue;
            
            int best_first_pos = 0;
            long long best_first_val = (long long)d[i][cycle[0]] + nodes[i].cost;
            int best_second_pos = 0;
            long long best_second_val = LLONG_MAX;

            int c_size = (int)cycle.size();
            for (int pos = 1; pos < c_size; pos++) {
                long long val = (long long)d[cycle[pos - 1]][i] + d[cycle[pos]][i] - d[cycle[pos - 1]][cycle[pos]] + nodes[i].cost;

                if (val < best_first_val)
                {
                    best_second_val = best_first_val;
                    best_second_pos = best_first_pos;
                    best_first_val = val;
                    best_first_pos = pos;
                } else if (val < best_second_val)
                {
                    best_second_val = val;
                    best_second_pos = pos;
                }
            }

            long long end_val = d[cycle[c_size - 1]][i] + nodes[i].cost;
            if (end_val < best_first_val)
                {
                    best_second_val = best_first_val;
                    best_second_pos = best_first_pos;
                    best_first_val = end_val;
                    best_first_pos = c_size;
                } else if (end_val < best_second_val)
                {
                    best_second_val = end_val;
                    best_second_pos = c_size;
                }

            ranking.push_back({i, best_first_pos, best_first_val, best_second_pos, best_second_val});
        }

        double best_score = -__DBL_MAX__;
        int best_node = 0;
        int best_pos = 0;

        for(RegretRankingEntry entry: ranking) {
            double score = w1 * (double)(entry.second_val - entry.first_val) - w2 * (double)entry.first_val;
            
            if (score > best_score)
            {
                best_score = score;
                best_node = entry.node;
                best_pos = entry.first_pos;
            }
            
        }
        
        
        cycle.insert(cycle.begin() + best_pos, best_node);
        used[best_node] = true;
    }

    return cycle;
}


vector<int> greedy_cycle_regret(int start_node, int k, const vector<vector<int>>& d, const vector<Node>& nodes, double w1, double w2) {
    struct RegretRankingEntry {
            int node;
            int first_pos;
            long long first_val;
            int second_pos;
            long long second_val;
        };

    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.reserve(k);

    cycle.push_back(start_node);
    used[start_node] = true;

    int best_second = -1;
    long long best_delta = LLONG_MAX;

    // add second node to form first edge
    for (int cand = 0; cand < n; cand++)
    {
        if (used[cand])
            continue;
        long long delta = (long long)d[start_node][cand] + nodes[cand].cost;
        if (delta < best_delta)
        {
            best_delta = delta;
            best_second = cand;
        }
    }

    cycle.push_back(best_second);
    used[best_second] = true;


    while ((int)cycle.size() < k)
    {
        
        vector<RegretRankingEntry> ranking;

        for (int i = 0; i < n; i++)
        {
            if (used[i]) continue;
            
            int best_first_pos = 0;
            long long best_first_val = LLONG_MAX;
            int best_second_pos = 0;
            long long best_second_val = LLONG_MAX;

            int c_size = (int)cycle.size();
            for (int pos = 1; pos <= c_size; pos++) {
                long long val = (long long)d[cycle[pos - 1]][i] + d[cycle[pos % c_size]][i] + nodes[i].cost;
                if (c_size > 2)
                {
                    val -= d[cycle[pos - 1]][cycle[pos % c_size]];
                }

                if (val < best_first_val)
                {
                    best_second_val = best_first_val;
                    best_second_pos = best_first_pos;
                    best_first_val = val;
                    best_first_pos = pos;
                } else if (val < best_second_val)
                {
                    best_second_val = val;
                    best_second_pos = pos;
                }
            }

            ranking.push_back({i, best_first_pos, best_first_val, best_second_pos, best_second_val});
        }

        double best_score = -__DBL_MAX__;
        int best_node = 0;
        int best_pos = 0;

        for(RegretRankingEntry entry: ranking) {
            double score = w1 * (double)(entry.second_val - entry.first_val) - w2 * (double)entry.first_val;
            
            if (score > best_score)
            {
                best_score = score;
                best_node = entry.node;
                best_pos = entry.first_pos;
            }
            
        }
        
        
        cycle.insert(cycle.begin() + best_pos, best_node);
        used[best_node] = true;
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


void print_stats(const string& heuristic_name, const vector<long long>& objectives) {
    if (objectives.empty()) {
        cerr << "Error: No objectives recorded for " << heuristic_name << ".\n";
        return;
    }

    long long min_obj = *min_element(objectives.begin(), objectives.end());
    long long max_obj = *max_element(objectives.begin(), objectives.end());

    long double sum_obj = accumulate(objectives.begin(), objectives.end(), (long double)0.0);
    long double avg_obj = sum_obj / objectives.size();

    cout << "\n " << heuristic_name << " Stats\n";
    cout << "  Min Objective: " << min_obj << "\n";
    cout << "  Max Objective: " << max_obj << "\n";
    cout << "  Avg Objective: " << avg_obj << "\n";
    cout << "------------------------------------------\n";
}


void export_tour_txt(const string& filename, const vector<int>& tour) {
    ofstream outfile(filename);
    if (outfile.is_open()) {
        for (size_t i = 0; i < tour.size(); ++i) {
            outfile << tour[i];
            if (i < tour.size() - 1) {
                outfile << "\n";
            }
        }
        outfile << "\n";
        outfile.close();
        cout << "  > Exported best tour indices to " << filename << " (TXT file).\n";
    } else {
        cerr << "  > ERROR: Unable to open file " << filename << " for writing.\n";
    }
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

    // 1. Random Solutions
    long long best_random_obj = LLONG_MAX;
    vector<int> best_random_tour;
    vector<long long> random_objectives;

    // Generate N_SOL random solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        auto tour = random_solution(n, k);
        long long obj = tour_objective(tour, d, nodes);
        random_objectives.push_back(obj);

        if(obj < best_random_obj){
            best_random_obj = obj;
            best_random_tour = tour;
        }
    }
    print_stats("Random Solution", random_objectives);
    export_tour_svg("best_random_tour.svg", best_random_tour, nodes);
    export_tour_txt("best_random_tour.txt", best_random_tour);

    // 2. NN End Solutions
    long long best_nn_end_obj = LLONG_MAX;
    vector<int> best_nn_end_tour;
    vector<long long> nn_end_objectives;

    // Generate N_SOL nn_end solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = nn_end(start, k, d, nodes);
        long long obj = tour_objective(tour, d, nodes);
        nn_end_objectives.push_back(obj);

        if(obj < best_nn_end_obj){
            best_nn_end_obj = obj;
            best_nn_end_tour = tour;
        }
    }
    print_stats("Nearest Neighbor (End)", nn_end_objectives);
    export_tour_svg("best_end_nn_tour.svg", best_nn_end_tour, nodes);
    export_tour_txt("best_end_nn_tour.txt", best_nn_end_tour);

    // 3. NN Any Position Solutions
    long long best_nn_any_obj = LLONG_MAX;
    vector<int> best_nn_any_tour;
    vector<long long> nn_anypos_objectives;

    // Generate N_SOL NN any-position and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = nn_anypos(start, k, d, nodes);
        long long obj = tour_objective(tour, d, nodes);
        nn_anypos_objectives.push_back(obj);

        if(obj < best_nn_any_obj){
            best_nn_any_obj = obj;
            best_nn_any_tour = tour;
        }
    }
    print_stats("Nearest Neighbor (Any Position)", nn_anypos_objectives);
    export_tour_svg("best_anypos_nn_tour.svg", best_nn_any_tour, nodes);
    export_tour_txt("best_anypos_nn_tour.txt", best_nn_any_tour);

    // 4. Greedy Cycle Solutions
    long long best_greedy_obj = LLONG_MAX;
    vector<int> best_greedy_tour;
    vector<long long> greedy_cycle_objectives;

    // Generate N_SOL Greedy cycle solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = greedy_cycle(start, k, d, nodes);
        long long obj = tour_objective(tour, d, nodes);
        greedy_cycle_objectives.push_back(obj);

        if(obj < best_greedy_obj){
            best_greedy_obj = obj;
            best_greedy_tour = tour;
        }
    }
    print_stats("Greedy Cycle", greedy_cycle_objectives);
    export_tour_svg("best_greedy_cycle_tour.svg", best_greedy_tour, nodes);
    export_tour_txt("best_greedy_cycle_tour.txt", best_greedy_tour);


    // 5. NN Any Position Regret Solutions
    long long best_nn_any_regret_obj = LLONG_MAX;
    vector<int> best_nn_any_regret_tour;
    vector<long long> nn_anypos_regret_objectives;

    // Generate N_SOL NN any-position _regret and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = nn_anypos_regret(start, k, d, nodes, 1.0, 0.0);
        long long obj = tour_objective(tour, d, nodes);
        nn_anypos_regret_objectives.push_back(obj);

        if(obj < best_nn_any_regret_obj){
            best_nn_any_regret_obj = obj;
            best_nn_any_regret_tour = tour;
        }
    }
    print_stats("Nearest Neighbor (Any Position Regret)", nn_anypos_regret_objectives);
    export_tour_svg("best_anypos_nn_regret_tour.svg", best_nn_any_regret_tour, nodes);
    export_tour_txt("best_anypos_nn_regret_tour.txt", best_nn_any_regret_tour);


    // 6. Greedy Cycle Regret Solutions
    long long best_greedy_regret_obj = LLONG_MAX;
    vector<int> best_greedy_regret_tour;
    vector<long long> greedy_cycle_regret_objectives;

    // Generate N_SOL Greedy cycle regret solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = greedy_cycle_regret(start, k, d, nodes, 1.0, 0.0);
        long long obj = tour_objective(tour, d, nodes);
        greedy_cycle_regret_objectives.push_back(obj);

        if(obj < best_greedy_regret_obj){
            best_greedy_regret_obj = obj;
            best_greedy_regret_tour = tour;
        }
    }
    print_stats("Greedy Cycle Regret", greedy_cycle_regret_objectives);
    export_tour_svg("best_greedy_cycle_regret_tour.svg", best_greedy_regret_tour, nodes);
    export_tour_txt("best_greedy_cycle_regret_tour.txt", best_greedy_regret_tour);

     // 7. NN Any Position Regret Weighted Solutions
    long long best_nn_any_regret_weighted_obj = LLONG_MAX;
    vector<int> best_nn_any_regret_weighted_tour;
    vector<long long> nn_anypos_regret_weighted_objectives;

    // Generate N_SOL NN any-position _regret _weighted and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = nn_anypos_regret(start, k, d, nodes, 0.5, 0.5);
        long long obj = tour_objective(tour, d, nodes);
        nn_anypos_regret_weighted_objectives.push_back(obj);

        if(obj < best_nn_any_regret_weighted_obj){
            best_nn_any_regret_weighted_obj = obj;
            best_nn_any_regret_weighted_tour = tour;
        }
    }
    print_stats("Nearest Neighbor (Any Position Regret Weighted)", nn_anypos_regret_weighted_objectives);
    export_tour_svg("best_anypos_nn_regret_weighted_tour.svg", best_nn_any_regret_weighted_tour, nodes);
    export_tour_txt("best_anypos_nn_regret_weighted_tour.txt", best_nn_any_regret_weighted_tour);


    // 8. Greedy Cycle Regret Weighted Solutions
    long long best_greedy_regret_weighted_obj = LLONG_MAX;
    vector<int> best_greedy_regret_weighted_tour;
    vector<long long> greedy_cycle_regret_weighted_objectives;

    // Generate N_SOL Greedy cycle regret _weighted solutions and keep the best
    for(int t=0;t<N_SOL;t++){
        int start = t % n;
        auto tour = greedy_cycle_regret(start, k, d, nodes, 0.5, 0.5);
        long long obj = tour_objective(tour, d, nodes);
        greedy_cycle_regret_weighted_objectives.push_back(obj);

        if(obj < best_greedy_regret_weighted_obj){
            best_greedy_regret_weighted_obj = obj;
            best_greedy_regret_weighted_tour = tour;
        }
    }
    print_stats("Greedy Cycle Regret Weighted", greedy_cycle_regret_weighted_objectives);
    export_tour_svg("best_greedy_cycle_regret_tour.svg", best_greedy_regret_weighted_tour, nodes);
    export_tour_txt("best_greedy_cycle_regret_tour.txt", best_greedy_regret_weighted_tour);

    return 0;
}