"""
Script to extract and load GTFS data from DMRC_GTFS.zip and populate the database with real stops/routes/trips.
"""
import os
from gtfs_processor import GTFSProcessor
import zipfile

def main():
    gtfs_zip_path = os.path.abspath("C:/Hackathon Projects/SIH Project/DMRC_GTFS.zip")
    gtfs_extract_dir = os.path.abspath("C:/Hackathon Projects/SIH Project/gtfs_data/")

    print(f"Extracting GTFS zip from: {gtfs_zip_path}")
    with zipfile.ZipFile(gtfs_zip_path, 'r') as zip_ref:
        zip_ref.extractall(gtfs_extract_dir)
    print(f"Extracted to: {gtfs_extract_dir}")

    processor = GTFSProcessor()
    print("Loading GTFS files into database...")
    loaded = processor.load_gtfs_files(gtfs_extract_dir)
    if loaded:
        print("GTFS files loaded successfully. Data source: realtime")
        # Save to database, overwriting any mock data
        saved = processor.save_to_database()
        if saved:
            print("GTFS data saved to database.")
        else:
            print("Failed to save GTFS data to database.")
    else:
        print("Failed to load GTFS files. Data source: mock")

if __name__ == "__main__":
    main()