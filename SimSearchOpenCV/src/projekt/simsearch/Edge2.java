package projekt.simsearch;

class Edge2 {
    public Edge2(int to, long reduced_cost, long residual_capacity) {
        _to = to;
        _reduced_cost = reduced_cost;
        _residual_capacity = residual_capacity;
    }

    int _to;
    long _reduced_cost;
    long _residual_capacity;
}