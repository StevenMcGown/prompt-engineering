from openai import OpenAI
import os

# Get the OpenAI API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the quadratic equation
equation = "5x^2 + 6x + 1 = 0"

# First Prompt: Solve the equation
solve_prompt = f"""
You are a helpful assistant and an expert in solving quadratic equations. For the given quadratic equation, solve it step-by-step using the appropriate method. Follow these instructions:

1. **Check the equation**: Ensure the equation is in standard form ax^2 + bx + c = 0. If not, rewrite it in standard form.
2. **Identify coefficients**: Determine the values of a, b, and c.
3. **Choose the solving method**:
   - If the equation can be factored easily, solve by factoring.
   - Otherwise, use the quadratic formula: 
     x = (-b ± √(b^2 - 4ac)) / 2a
4. **Calculate the discriminant (b^2 - 4ac)**:
   - If the discriminant is positive, solve for two real solutions.
   - If the discriminant is zero, solve for the one real solution.
   - If the discriminant is negative, solve for two complex solutions.
5. **Solve for x**: Complete the calculations step-by-step.
6. **Double-check the solution**: Verify the solution by substituting the values back into the original equation.
7. **Print the answer**: Output the solution like so: Answer: x = -0.2 or x = -1

Now, solve the following quadratic equation step-by-step.:
{equation}

Your answer should only contain the answer in the requsted format and you should NOT try to style it with Latex either!
"""

# First API call to solve the equation
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that provides step-by-step mathematical solutions."},
        {"role": "user", "content": solve_prompt}
    ],
    max_tokens=1000,
    temperature=0
)

# Extract and print the solution from the first call
solution = response.choices[0].message.content.strip()

# Second Prompt: Validate and enforce proper format
format_validation_prompt = f"""
You are a helpful assistant and an expert in formatting quadratic equation solutions. The solution to the quadratic equation must be presented as:

Answer: x = [value1] or x = [value2]

Check if the following solution follows this format. Do not verify its mathematical correctness.
If it is in the right format, your output should ONLY contain the solution. Otherwise, if it is incorrectly formatted, you should return -1

Solution to Check:
{solution}
"""

# Second API call to validate the format
validation_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that verifies solution formats."},
        {"role": "user", "content": format_validation_prompt}
    ],
    max_tokens=100,
    temperature=0
)

# Extract and print the validation result from the second call
validation_result = validation_response.choices[0].message.content.strip()
print(validation_result)
