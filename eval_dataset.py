import json
import random
from datetime import datetime, timedelta
import os

# New pools (different from dataset.py)
first_names = [
    "Arjun", "Meera", "Isha", "Vikram", "Tara", "Karan", "Leena", "Dev", "Nisha", "Rohan"
]
last_names = [
    "Mehta", "Chopra", "Singh", "Bose", "Gupta", "Reddy", "Verma", "Kapoor", "Nair", "Joshi"
]

institutes = [
    "Pioneer Institute", "Horizon College", "Maplewood School", "Crestview University", "Lakeside Academy"
]

payment_types = [
    "Registration Fee",
    "Seminar Fee",
    "Hostel Deposit",
    "Workshop Fee",
    "Lab Equipment Fee",
    "Sports Membership",
]

# Longer templates: 3 multi-line email templates and 3 SMS-style templates
templates = [
    # Email template 1 (receipt + breakdown + contact)
    """Subject: Payment Receipt Confirmation – {payment}

Dear {name},

This is to confirm that we have received your payment of INR {amount} on {date} towards {payment}. The transaction reference is {txn} and your roll number is {roll}.

Payment details:
- Student: {name}
- Roll number: {roll}
- Amount received: INR {amount}
- Payment type: {payment}
- Transaction ID: {txn}
- Date: {date}

If you have any queries regarding this payment, please reply to this email or contact the accounts office at {institute}. Thank you for your prompt payment.

Best regards,
{institute} Accounts Team""",

    # Email template 2 (formal acknowledgement with reference and note)
    """Subject: Acknowledgement of Payment Received

Hello {name},

We hereby acknowledge receipt of INR {amount} on {date} for {payment}. Your transaction ID is {txn} and the payment has been credited against roll number {roll}.

Please retain this acknowledgement for your records. This payment will reflect in your student account within 24 hours. If the amount does not match or you believe there is an error, contact us quoting the transaction ID above.

Regards,
Finance Department
{institute}""",

    # Email template 3 (detailed multi-line, includes note and thanks)
    """Subject: Confirmation: Payment Received (Ref: {txn})

Dear {name},

This message confirms that we have successfully processed your payment of INR {amount} towards {payment} on {date}. Transaction reference: {txn}. Your roll number on file is {roll}.

Details:
  • Payer: {name}
  • Roll No.: {roll}
  • Amount: INR {amount}
  • Payment Purpose: {payment}
  • Transaction ID: {txn}
  • Date: {date}

Thank you for your payment. If you require an official receipt or have further questions, please contact the {institute} accounts office.

Sincerely,
Accounts Office - {institute}""",

    # SMS template 1 (short but descriptive)
    "{name}: INR {amount} received for {payment} on {date}. Txn {txn}. Roll {roll}. {institute}",

    # SMS template 2 (compact acknowledgement)
    "Payment of Rs.{amount} for {payment} received from {name} (Roll {roll}) on {date}. Ref:{txn}",

    # SMS template 3 (notification style with brief instructions)
    "Received: Rs {amount} for {payment} by {name} on {date}. TxnID:{txn}. Keep this ref. {institute}"
]


def generate_transaction_id():
    prefix = f"TX{random.randint(100, 999)}"
    mid = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=6))
    return f"{prefix}{mid}"


def generate_roll_number():
    return f"RN{random.randint(1000, 9999)}"


def generate_date():
    # random date in 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    delta = (end_date - start_date).days
    d = start_date + timedelta(days=random.randint(0, delta))
    return d.strftime("%B %d, %Y")


def generate_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        amount = random.choice([3000, 4500, 6000, 7500, 9000, 11000, 13000])
        roll = generate_roll_number()
        date = generate_date()
        txn = generate_transaction_id()
        payment = random.choice(payment_types)
        institute = random.choice(institutes)

        template = random.choice(templates)
        message = template.format(
            name=name,
            amount=amount,
            roll=roll,
            date=date,
            txn=txn,
            payment=payment,
            institute=institute,
        )

        input_text = f"Extract payment details: {message}"
        target_text = (
            f"student_name: {name} | "
            f"amount: {amount} | "
            f"roll_number: {roll} | "
            f"date: {date} | "
            f"transaction_id: {txn} | "
            f"payment_name: {payment} | "
            f"institute: {institute}"
        )

        data.append({"input_text": input_text, "target_text": target_text})

    return data


def main():
    print("🔄 Generating evaluation dataset (1000 samples)...")
    ds = generate_dataset(num_samples=1000)
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fee_messages_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)
    print(f"✅ Generated {len(ds)} samples")
    print(f"💾 Saved to: {out_path}")


if __name__ == "__main__":
    main()
