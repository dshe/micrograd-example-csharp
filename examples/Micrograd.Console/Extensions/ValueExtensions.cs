using Spectre.Console;
using System.Text;
namespace Micrograd.Console.Extensions;

/// <summary>
/// The <see cref="Value"/> extensions.
/// </summary>
public static class ValueExtensions
{
    /// <summary>
    /// Prints tree view of the Value.
    /// </summary>
    /// <param name="value">The input node.</param>
    public static void PrintAsTree(this Value value)
    {
        Tree root = NewMethod();

        // Create root node because tree is not inherited from Node type ^_^
        TreeNode rootNode = root.AddNode(GetFormattedTreeLabel(value));

        PopulateTree(rootNode, value);

        AnsiConsole.Write(root);

        static void PopulateTree(TreeNode node, Value value)
        {
            TreeNode nextNode = string.IsNullOrEmpty(value.Operation) ? node : node.AddNode($"[red]({value.Operation})[/]");
            foreach (Value child in value.Children)
            {
                string formattedValue = GetFormattedTreeLabel(child);
                TreeNode childNode = nextNode.AddNode(formattedValue);
                PopulateTree(childNode, child);
            }
        }
    }

    private static Tree NewMethod() => new("Tree View:");

    /// <summary>
    /// Gets the formatted label containing value, gradient and label.
    /// </summary>
    /// <param name="value">The input node.</param>
    /// <returns>The formatted label.</returns>
    private static string GetFormattedTreeLabel(Value value)
    {
        StringBuilder sb = new($"{value} || [yellow]grad={value.Gradient:0.#######}[/]");
        if (value.Label != null)
            sb.Append($" || [green]{value.Label}[/]");
        return sb.ToString();
    }
}
