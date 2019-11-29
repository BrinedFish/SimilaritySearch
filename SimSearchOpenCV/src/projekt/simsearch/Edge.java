package projekt.simsearch;


/**
 * @author Telmo Menezes (telmo@telmomenezes.com)
 *
 */
class Edge {
    public Edge(int to, long cost) {
        _to = to;
        _cost = cost;
    }

    int _to;
    long _cost;
}