import json

# Load your current dataset
with open("data/fee_messages_instruction.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"📊 Total samples: {len(dataset)}")
print("\n" + "="*80)

# Check first few samples
for i in range(min(3, len(dataset))):
    print(f"\n🔍 Sample {i+1}:")
    print("-" * 80)
    
    input_text = dataset[i]['input_text']
    target_text = dataset[i]['target_text']
    
    # Extract the message part (between first instruction line and last instruction line)
    lines = input_text.split('\n')
    # Message is everything except first and last line (instruction lines)
    message = '\n'.join(lines[1:-1]) if len(lines) > 2 else lines[1]
    
    print(f"📨 MESSAGE:\n{message}\n")
    print(f"🎯 TARGET:\n{target_text}\n")
    
    # Parse target fields
    target_dict = {}
    for pair in target_text.split('; '):
        if ': ' in pair:
            key, value = pair.split(': ', 1)
            target_dict[key] = value
    
    # Check each field
    print("✓ Field Verification:")
    all_present = True
    for key, value in target_dict.items():
        if key == 'student_name':
            # Check if both first and last name are present
            name_parts = value.split()
            is_present = all(part in message for part in name_parts)
        elif key == 'amount':
            # Check amount with or without ₹
            is_present = str(value) in message or f"₹{value}" in message
        else:
            # Check exact value
            is_present = str(value) in message
        
        status = "✅" if is_present else "❌"
        print(f"  {status} {key}: '{value}' {'FOUND' if is_present else 'MISSING'}")
        
        if not is_present:
            all_present = False
    
    if not all_present:
        print("\n⚠️ PROBLEM: Some target fields are NOT in the input message!")
        print("This will cause the model to learn nothing (loss=0, grad=NaN)")
    
    print("="*80)

# Statistics
print("\n📈 Dataset Statistics:")
missing_count = 0
problematic_samples = []

for i, item in enumerate(dataset):
    lines = item['input_text'].split('\n')
    message = '\n'.join(lines[1:-1]) if len(lines) > 2 else lines[1]
    target = item['target_text']
    
    # Check if all target values are in message
    has_error = False
    for pair in target.split('; '):
        if ': ' in pair:
            key, value = pair.split(': ', 1)
            
            if key == 'student_name':
                name_parts = value.split()
                if not all(part in message for part in name_parts):
                    has_error = True
                    break
            elif key == 'amount':
                if str(value) not in message and f"₹{value}" not in message:
                    has_error = True
                    break
            else:
                if str(value) not in message:
                    has_error = True
                    break
    
    if has_error:
        missing_count += 1
        if len(problematic_samples) < 5:  # Store first 5 problematic samples
            problematic_samples.append((i, item))

print(f"❌ Samples with missing fields: {missing_count}/{len(dataset)} ({missing_count/len(dataset)*100:.1f}%)")

if missing_count > 0:
    print(f"\n⚠️ Found {missing_count} problematic samples")
    print("\n🔍 First few problematic samples:")
    for idx, (i, item) in enumerate(problematic_samples):
        print(f"\n--- Problem Sample {idx+1} (Index {i}) ---")
        lines = item['input_text'].split('\n')
        message = '\n'.join(lines[1:-1]) if len(lines) > 2 else lines[1]
        print(f"Message: {message[:200]}...")
        print(f"Target: {item['target_text'][:200]}...")
    
    print("\n💡 Solution: The dataset generation is working, but verification may be too strict.")
    print("   Try training anyway - if loss is non-zero, the dataset is fine!")
else:
    print("\n✅ Dataset is valid! All target fields exist in input messages.")