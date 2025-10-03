import boto3
import os
from dotenv import load_dotenv

load_dotenv()

REGION = "eu-north-1"
GUARDRAIL_ID = "cv32rc6w1uar"  # Just the ID, not ARN

def test_bedrock_access():
    """Test basic Bedrock access"""
    try:
        bedrock = boto3.client("bedrock", region_name=REGION)
        response = bedrock.list_guardrails()
        print("✅ Basic Bedrock access works")
        print(f"Found {len(response.get('guardrails', []))} guardrails")
        return True
    except Exception as e:
        print(f"❌ Basic Bedrock access failed: {e}")
        return False

def test_guardrail_get():
    """Test getting specific guardrail"""
    try:
        bedrock = boto3.client("bedrock", region_name=REGION)
        
        # Try with just ID
        response = bedrock.get_guardrail(
            guardrailIdentifier=GUARDRAIL_ID
        )
        print(f"✅ Guardrail accessible: {response['name']}")
        print(f"   Status: {response['status']}")
        print(f"   Version: {response['version']}")
        return response
    except Exception as e:
        print(f"❌ Guardrail get failed: {e}")
        return None

def test_guardrail_apply():
    """Test applying guardrail"""
    try:
        bedrock_rt = boto3.client("bedrock-runtime", region_name=REGION)
        
        response = bedrock_rt.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion="DRAFT",  # Try DRAFT first
            source="INPUT",
            content=[{
                "text": {
                    "text": "Hello, how are you?"
                }
            }]
        )
        print("✅ Guardrail apply works!")
        print(f"   Action: {response.get('action')}")
        return response
    except Exception as e:
        print(f"❌ Guardrail apply failed: {e}")
        
        # Try with version 1
        try:
            response = bedrock_rt.apply_guardrail(
                guardrailIdentifier=GUARDRAIL_ID,
                guardrailVersion="1",
                source="INPUT",
                content=[{
                    "text": {
                        "text": "Hello, how are you?"
                    }
                }]
            )
            print("✅ Guardrail apply works with version 1!")
            print(f"   Action: {response.get('action')}")
            return response
        except Exception as e2:
            print(f"❌ Guardrail apply with version 1 also failed: {e2}")
        
        return None

if __name__ == "__main__":
    print("=== Minimal Bedrock Guardrail Test ===")
    print(f"Region: {REGION}")
    print(f"Guardrail ID: {GUARDRAIL_ID}")
    
    print("\n1. Testing basic Bedrock access...")
    if not test_bedrock_access():
        print("Fix basic Bedrock permissions first")
        exit(1)
    
    print("\n2. Testing guardrail get...")
    guardrail_info = test_guardrail_get()
    if not guardrail_info:
        print("Fix guardrail get permissions")
        exit(1)
    
    print("\n3. Testing guardrail apply...")
    result = test_guardrail_apply()
    if result:
        print("\n🎉 Everything works! Update your main script accordingly.")
    else:
        print("\n❌ Guardrail apply still not working. Check runtime permissions.")