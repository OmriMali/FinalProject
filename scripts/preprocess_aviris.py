from src import io
from src.preprocessing.hsi import trim_borders, crop_hsi_sections, filter_spectral_bands


def main():

    # ===== Parameters =====
    filter_spectra = True
    trim_border = True
    crop_to_sections = True

    bad_spectral_ranges = [(104, 108), (150, 163)]
    bad_spectral_bands = [220]
    section_size = (256, 256)

    # ===== Paths =====
    raw_dir = r"data\raw"
    processed_dir = r"data\processed"
    
    # ===== Scenes =====
    scenes = ["JasperRidge", "MoffetField", "Cuprite"]

    # ===== Preprocess =====
    for scene in scenes:
        hsi = io.load_aviris_folder(rf"{raw_dir}\{scene}")

        if filter_spectra:
            hsi = filter_spectral_bands(hsi,
                                        remove_ranges=bad_spectral_ranges,
                                        remove_bands=bad_spectral_bands)
        
        if trim_border:
            hsi = trim_borders(hsi, black_value=-50)

        io.save_hsi(hsi, rf"{processed_dir}\{scene}", scene)

        if crop_to_sections:
            sections = crop_hsi_sections(hsi, section_size)
            for sec in sections:
                io.save_hsi(sec, fr"{processed_dir}\{scene}\sections", f"{scene}_r{sec.metadata.section_row}_c{sec.metadata.section_col}")

        
if __name__ == "__main__":
    main()