namespace Micrograd.NN;

public abstract class Module
{
    public abstract Value[] GetParameters();

    public void ZeroGrad()
    {
        foreach (Value p in GetParameters())
            p.ZeroGrad();
    }
}
