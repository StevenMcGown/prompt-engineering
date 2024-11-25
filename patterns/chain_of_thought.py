from openai import OpenAI
import os


# Get the OpenAI API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the chain-of-thought budgeting scenario
prompt = """
You have a monthly income of $8000 and the following list of monthly expenses:
- Rent: $1,200
- Electricity: $150
- Water: $55
- Phone: $89
- Internet: $35
- Groceries: $400
- Gym: $40
- Transportation: $50
- Miscellaneous: $100

You also have the following debts to pay:
- Loan 1:	$6,224.33	3.760%
- Loan 2:	$6,115.95	4.450%
- Loan 3:	$1,087.50	4.450%
- Loan 4:	$4,753.65	5.050%
- Loan 5:	$7,706.00	4.530%
- Loan 6:	$2,501.80	2.750%
- Loan 7:	$2,501.80	2.750%
- Loan 8:   $2,253.00   4.900%
- Loan 9:   $7,980.00   28.00% (Will start accruing interest in MAY)

You want to save as much money as you can to pay off the debt, but you want to make sure to pay off loan 9 before it starts accruing interest. Break down each step and calculate any adjustments needed.

Before you do any calculations, determine how much will need to allocate pay on Loan 9 before it starts to accrue interest. Right now, it's November.

1. Keep track of what month it is.
2. Calculate total expenses and determine remaining balance after expenses.
3. Take the remaining balance from the debt left over from the previous month.

"""

# Make the API call with the chain-of-thought prompt
response = client.chat.completions.create(model="gpt-4-turbo",
messages=[
    {"role": "system", "content": "You are a helpful assistant that provides step-by-step budget planning."},
    {"role": "user", "content": prompt}
],
max_tokens=200,
temperature=0.5)

# Print the generated response
print(response.choices[0].message.content.strip())
