from openai import OpenAI
import os
from IPython.display import display, Markdown

# Get the OpenAI API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

problem = "Find the partial differential equation of u = f(x2 - y2)."

# Define the Meta Prompt
meta_prompt = f"""
You are a structured assistant specialized in solving advanced mathematical problems, particularly Partial Differential Equations (PDEs), using a systematic and syntax-driven approach.

### Key Notes:
- Partial Differential Equations (PDEs) relate a function of multiple variables to its partial derivatives. They describe phenomena like sound, heat, and fluid flow.
- PDEs are classified by their order (highest derivative), degree, and discriminant (B^2 - AC), which determines if they are elliptic, parabolic, or hyperbolic.
- Methods to solve PDEs include separation of variables, substitution, or change of variables.

### Expected Response Format:
Your response must adhere to the following structure:

1. Identify Problem Type:
   - Clearly classify the problem:
     - **Order**: Specify whether the PDE is first-order or second-order based on the highest derivative.
     - **Degree**: Indicate the degree of the highest derivative in the PDE.
     - **Classification** (for second-order PDEs): Use the discriminant \( B^2 - AC \) to classify as:
       - **Elliptic**: If \( B^2 - AC < 0 \).
       - **Parabolic**: If \( B^2 - AC = 0 \).
       - **Hyperbolic**: If \( B^2 - AC > 0 \).

   Example:
   Problem Type: First-order linear PDE.

2. Selected Method:
   - Specify the method or approach used to solve the PDE. Examples include:
     - Differentiation and substitution
     - Separation of variables
     - Change of variables
     - Fourier or Laplace transforms
   - Briefly justify why this method is appropriate for the problem.

   Example:
   Selected Method: Differentiation with respect to \( x \) and \( y \), followed by elimination of the arbitrary function \( f \).

3. Step-by-Step Solution:
   - Provide a detailed breakdown:
     - **Step 1**: Write the given function and compute the required partial derivatives.
     - **Step 2**: Combine or manipulate the derivatives to eliminate arbitrary functions or constants.
     - **Step 3**: Simplify the resulting expression into a standard PDE form.
   - Include mathematical expressions and reasoning for each step.

   Example:
   Step 1: Differentiate \( u = f(x^2 - y^2) \) with respect to \( x \) and \( y \):
   \[
   \frac{{\partial u}}{{\partial x}} = 2x f'(x^2 - y^2), \quad \frac{{\partial u}}{{\partial y}} = -2y f'(x^2 - y^2).
   \]

   Step 2: Eliminate \( f'(x^2 - y^2) \):
   \[
   \frac{{\frac{{\partial u}}{{\partial x}}}}{{\frac{{\partial u}}{{\partial y}}}} = -\frac{{x}}{{y}}.
   \]

   Step 3: Derive the PDE:
   \[
   y \frac{{\partial u}}{{\partial x}} + x \frac{{\partial u}}{{\partial y}} = 0.
   \]

4. Final Answer:
   - Present the final PDE in this exact format:
     \[
     [Dependent Variable(s)] = [Derived PDE]
     \]

   Example:
   Final Answer:
   \[
   y \frac{{\partial u}}{{\partial x}} + x \frac{{\partial u}}{{\partial y}} = 0
   \]

### Problem:
{problem}

Solve the problem step-by-step and provide the solution in the specified format.
"""

# First API call to solve the equation
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that provides step-by-step mathematical solutions."},
        {"role": "user", "content": meta_prompt}
    ],
    max_tokens=1000,
    temperature=0
)

# Extract and print the solution from the first call
solution = response.choices[0].message.content.strip()
# Second Prompt: Validate and enforce proper format

print(solution)

format_validation_prompt = f"""
You are a helpful assistant and an expert in formatting partial differential equation solutions. The solution to the partial differential equation must be presented as:

[Dependent Variable(s)] = [Answer]

Check if the following solution follows this format. Do not verify its mathematical correctness. It's important that you retain the answer the way it's presented in the solution.
If it is in the right format, your output should ONLY contain the solution.

Solution to Check:
{solution}
"""

# Second API call to validate the format
validation_response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that verifies solution formats."},
        {"role": "user", "content": format_validation_prompt}
    ],
    max_tokens=1000,
    temperature=0
)

# Extract and print the validation result from the second call
validation_result = validation_response.choices[0].message.content.strip()
print(validation_result)

