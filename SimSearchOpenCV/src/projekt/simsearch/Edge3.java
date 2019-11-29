package projekt.simsearch;

class Edge3 {
    public Edge3() {
        _to = 0;
        _dist = 0;
    }

    public Edge3(int to, long dist) {
        _to = to;
        _dist = dist;
    }

    int _to;
    long _dist;
}