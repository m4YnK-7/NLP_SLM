import json
import random
from datetime import datetime, timedelta

# Sample data pools
first_names = ["Aditi", "Kristen", "Nathan", "Priya", "Rahul", "Sarah", "John", "Emily", "Michael", "Anjali"]
last_names = ["Rao", "Wells", "Martinez", "Sharma", "Kumar", "Johnson", "Smith", "Brown", "Davis", "Patel"]

institutes = [
    "SkillUp Academy",
    "NextGen Learning Center", 
    "Bright Minds Academy",
    "Excellence Institute",
    "Future Leaders College"
]

payment_types = [
    "Tuition Fee",
    "Hostel Rent",
    "Exam Fee",
    "Library Fee",
    "Lab Fee",
    "Sports Fee",
    "AI Workshop Enrollment",
    "Annual Fee"
]

# Message templates
templates = [
    "Payment of ₹{amount} received from {name} (Roll No: {roll}) for {payment} on {date}. Txn ID: {txn}. - {institute}",
    "Hi {name}, we've received ₹{amount} towards {payment} on {date}. Ref: {txn}, Roll: {roll}. - {institute}",
    "₹{amount} received from {name} ({roll}) for {payment}. Paid on {date}. Txn#: {txn}. - {institute}",
    "Dear {name}, your payment of ₹{amount} for {payment} has been confirmed. Transaction ID: {txn}, Date: {date}, Roll Number: {roll}. - {institute}",
    """Subject: Payment Confirmation for {payment}
Dear {name},
We received your payment of ₹{amount} for {payment} on {date}.
Transaction ID: {txn}
Roll Number: {roll}
Thank you for your timely payment.
Regards, {institute} Accounts Office""",
    """Subject: Acknowledgment of Payment – {payment}
Hello {name},
Your payment of ₹{amount} for {payment} was processed on {date}.
Transaction ID: {txn}
Roll No: {roll}
Thank you.
{institute}""",
]

def generate_transaction_id():
    """Generate realistic transaction ID"""
    prefix = f"TXN{random.randint(1000, 9999)}"
    middle = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
    suffix = random.randint(1000, 9999)
    return f"{prefix}{middle}{suffix}"

def generate_roll_number():
    """Generate roll number"""
    return f"R{random.randint(1000, 9999)}"

def generate_date():
    """Generate random date in 2025"""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    random_days = random.randint(0, (end_date - start_date).days)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%B %d, %Y")

def generate_dataset(num_samples=2000):
    """Generate synthetic dataset"""
    dataset = []
    
    for _ in range(num_samples):
        # Generate data
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        amount = random.choice([5000, 8000, 10000, 12000, 15000, 20000, 25000])
        roll = generate_roll_number()
        date = generate_date()
        txn = generate_transaction_id()
        payment = random.choice(payment_types)
        institute = random.choice(institutes)
        
        # Create message using template
        template = random.choice(templates)
        message = template.format(
            name=name,
            amount=amount,
            roll=roll,
            date=date,
            txn=txn,
            payment=payment,
            institute=institute
        )
        
        # FIXED: Simpler instruction format
        input_text = f"Extract payment details: {message}"
        
        # FIXED: Use pipe separator and simpler format for better tokenization
        target_text = (
            f"student_name: {name} | "
            f"amount: {amount} | "
            f"roll_number: {roll} | "
            f"date: {date} | "
            f"transaction_id: {txn} | "
            f"payment_name: {payment} | "
            f"institute: {institute}"
        )
        
        dataset.append({
            "input_text": input_text,
            "target_text": target_text
        })
    
    return dataset

# Generate dataset
print("🔄 Generating synthetic dataset...")
dataset = generate_dataset(num_samples=2000)

# Save to file
import os
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "fee_messages_instruction.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"✅ Generated {len(dataset)} samples")
print(f"💾 Saved to: {output_path}")

# Print samples
print("\n📋 Sample entries:")
for i in range(2):
    sample = dataset[i]
    print(f"\n--- Sample {i+1} ---")
    print(f"Input: {sample['input_text'][:100]}...")
    print(f"Target: {sample['target_text'][:100]}...")

print("\n" + "="*80)
print("✅ Dataset ready! Run the training script next.")
print("="*80)