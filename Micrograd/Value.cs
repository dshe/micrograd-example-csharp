namespace Micrograd;

public class Value(double data, IEnumerable<Value>? children = null, string op = "") : IComparable<Value>
{
    public double Data { get; set; } = data;
    public string Label { get; set; } = "";
    public double Gradient { get; private set; } = 0;
    public string Operation { get; private set; } = op;
    public IEnumerable<Value> Children { get; private set; } = children ?? [];
    public static implicit operator Value(int other) => new(other);
    public static implicit operator Value(double other) => new(other);
    public static implicit operator Value(float other) => new(other);

    private Action _backward = () => { };

    public static Value operator +(Value left, Value right)
    {
        Value result = new(left.Data + right.Data, [left, right ], "+");
        result._backward = () =>
        {
            left.Gradient += result.Gradient;
            right.Gradient += result.Gradient;
        };
        return result;
    }

    public static Value operator *(Value left, Value right)
    {
        Value result = new(left.Data * right.Data, [left, right], "*");
        result._backward = () =>
        {
            left.Gradient += right.Data * result.Gradient;
            right.Gradient += left.Data * result.Gradient;
        };
        return result;
    }

    public static Value operator -(Value val) => val * (-1);
    public static Value operator -(Value left, Value right) => left + (-right);
    public static Value operator /(Value left, Value right) => left * right.Pow(-1);

    public Value Exp()
    {
        Value result = new(Math.Exp(this.Data), new[] { this }, "exp");
        result._backward = () =>
        {
            this.Gradient += result.Data * result.Gradient;
        };
        return result;
    }

    public Value Pow(int power)
    {
        Value result = new(Math.Pow(this.Data, power), [this], $"pow({power})");
        result._backward = () =>
        {
            this.Gradient += power * Math.Pow(this.Data, power - 1) * result.Gradient;
        };
        return result;
    }

    public Value Tanh()
    {
        Value result = new(Math.Tanh(this.Data), [this], "tanh");
        result._backward = () =>
        {
            this.Gradient += (1 - Math.Pow(result.Data, 2)) * result.Gradient;
        };
        return result;
    }

    public Value RelU()
    {
        Value result = new(Math.Max(0, this.Data), [this], "relu");
        result._backward = () =>
        {
            this.Gradient += (this.Data > 0 ? 1 : 0) * result.Gradient;
        };
        return result;
    }

    public void Backward()
    {
        List<Value> topology = [];
        HashSet<Value> visited = [];

        BuildTopology(this);

        this.Gradient = 1;

        foreach (Value val in Enumerable.Reverse(topology))
            val._backward.Invoke();

        void BuildTopology(Value value)
        {
            if (visited.Contains(value))
                return;

            visited.Add(value);
            foreach (Value child in value.Children)
                BuildTopology(child);

            topology.Add(value);
        }
    }

    public void ZeroGrad() => this.Gradient = 0;
    public override string ToString() => $"value={this.Data}";
    public int CompareTo(Value? other) => this.Data.CompareTo(other?.Data);
}
