"""
Script to fetch and process real-time vehicle positions from Delhi OTD using RealtimeArchiver.
"""
from realtime_archiver import RealtimeArchiver

def main():
    api_key = "L5jVvGl6iLEdSqnZG42pEm5LD94t1PYF"  # Your Delhi OTD API key
    archiver = RealtimeArchiver(api_key=api_key)

    print("Fetching real-time vehicle positions...")
    positions = archiver.fetch_realtime_data()
    if positions:
        print(f"Fetched {len(positions)} vehicle positions.")
        # Print a sample
        for pos in positions[:5]:
            print(pos)
    else:
        print("No vehicle positions fetched or an error occurred.")

if __name__ == "__main__":
    main()
