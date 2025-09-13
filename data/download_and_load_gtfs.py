"""
Script to download and load GTFS data for Delhi OTD using GTFSProcessor.
"""
from gtfs_processor import GTFSProcessor

def main():
    gtfs_url = "https://otd.delhi.gov.in/gtfs.zip"  # Official Delhi OTD GTFS feed
    processor = GTFSProcessor()

    print("Downloading GTFS data...")
    success = processor.download_gtfs_data(gtfs_url)
    if not success:
        print("Failed to download GTFS data.")
        return

    print("Loading GTFS files...")
    loaded = processor.load_gtfs_files("gtfs_data/")
    if loaded:
        print("GTFS files loaded successfully.")
    else:
        print("Failed to load GTFS files.")

    print(f"Data source: {processor.data_source}")

if __name__ == "__main__":
    main()
