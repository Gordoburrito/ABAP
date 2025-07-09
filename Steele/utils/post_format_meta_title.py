import re
from PIL import ImageFont, Image, ImageDraw

# Cache the font to avoid reloading it
_font = None

def get_font(size=20):
    """Get or load the font for width calculations."""
    global _font
    if _font is None:
        try:
            # Try to load Roboto font at 20px (Google's typical font)
            _font = ImageFont.truetype("Roboto.ttf", size)
        except IOError:
            try:
                # Try to load a similar font if Roboto isn't available
                _font = ImageFont.truetype("Arial.ttf", size)
            except IOError:
                # Fallback to default font if neither is available
                _font = ImageFont.load_default()
    return _font

def calculate_pixel_width(text):
    """Calculate the pixel width of text using Pillow."""
    font = get_font()
    # For newer Pillow versions
    if hasattr(font, "getlength"):
        return font.getlength(text)
    # For older Pillow versions
    else:
        return font.getsize(text)[0]

def normalize_brands(title):
    """Normalize brand names to proper case."""
    return re.sub(r'\bMOPAR\b', 'Mopar', title)

def extract_title_parts(title):
    """Split title into main content and category if present."""
    if "|" in title:
        main_part, category = title.split("|", 1)
        return main_part.strip(), category.strip()
    return title.strip(), ""

def extract_year_range(main_part):
    """Extract and format year range from the title."""
    year_pattern = r'(?:for\s+)?(\d{4}(?:\s*-\s*\d{4})?)'
    year_match = re.search(year_pattern, main_part)
    
    if not year_match:
        return ""
        
    year_range = year_match.group(1)
    # Ensure proper spacing around dash
    if "-" in year_range:
        start_year, end_year = re.split(r'\s*-\s*', year_range)
        return f"{start_year} - {end_year}"
    return year_range

def extract_makes(main_part):
    """Extract vehicle makes from the title."""
    make_patterns = [
        r'(?:for\s+)?\d{4}(?:\s*-\s*\d{4})?\s+(.*?)(?:\||$)',
        r'for\s+(.*?)(?:\||$)'
    ]
    
    for pattern in make_patterns:
        make_match = re.search(pattern, main_part)
        if make_match:
            makes_text = make_match.group(1).strip()
            # Replace "Mopar Cars" with full list
            if "Mopar Cars" in makes_text:
                makes_text = makes_text.replace("Mopar Cars", "Chrysler, DeSoto, Dodge, and Plymouth")
            
            # Extract individual makes
            return [make.strip() for make in re.split(r',|&|\band\b', makes_text) if make.strip()]
    
    return []

def extract_product_name(main_part):
    """Extract product name from the beginning of the title."""
    product_match = re.match(r'^(.*?)(?:for|\d{4}|\||$)', main_part)
    if product_match:
        return product_match.group(1).strip()
    return ""

def determine_vintage_type(year_range):
    """Determine if vehicles are vintage, classic, or both based on year range."""
    if not year_range:
        return ""
        
    years = re.findall(r'\d{4}', year_range)
    if not years:
        return ""
        
    start_year = int(years[0])
    end_year = int(years[-1]) if len(years) > 1 else start_year
    
    if end_year <= 1949:
        return "Vintage "
    elif start_year >= 1950:
        return "Classic "
    else:
        return "Vintage & Classic "

def format_makes_list(makes):
    """Format the list of makes into a readable string."""
    if not makes:
        return ""
    
    if len(makes) == 1:
        return makes[0]
    
    return ", ".join(makes[:-1]) + ", and " + makes[-1]

def shorten_title(title, max_pixels=580):
    """Apply progressive shortening strategies to fit the title within max_pixels."""
    applied_transforms = []
    
    if calculate_pixel_width(title) <= max_pixels:
        return title, applied_transforms
    
    # Step 1: Remove spaces between dashes in years
    new_title = re.sub(r'(\d{4})\s+-\s+(\d{4})', r'\1-\2', title)
    if new_title != title:
        applied_transforms.append("Removed spaces around dash in year range")
        title = new_title
        if calculate_pixel_width(title) <= max_pixels:
            return title, applied_transforms
    
    # Step 2: Replace "and" with "&"
    new_title = title.replace(" and ", " & ")
    if new_title != title:
        applied_transforms.append('Replaced "and" with "&"')
        title = new_title
        if calculate_pixel_width(title) <= max_pixels:
            return title, applied_transforms
    
    # Step 3: Remove the last comma in a list before "&"
    new_title = re.sub(r',\s+&', ' &', title)
    if new_title != title:
        applied_transforms.append("Removed comma before &")
        title = new_title
        if calculate_pixel_width(title) <= max_pixels:
            return title, applied_transforms
    
    # Step 4: Remove all commas and "&"
    new_title = re.sub(r',', '', title)
    new_title = new_title.replace(" & ", " ")
    if new_title != title:
        applied_transforms.append("Removed all commas and '&' symbols")
        title = new_title
        if calculate_pixel_width(title) <= max_pixels:
            return title, applied_transforms
    
    # Step 5: Replace "for" with " | "
    new_title = re.sub(r'\bfor\b', '|', title)
    if new_title != title:
        applied_transforms.append('Replaced "for" with "|"')
        title = new_title
        if calculate_pixel_width(title) <= max_pixels:
            return title, applied_transforms
    
    # Step 6: Remove the category portion completely
    if '|' in title:
        parts = title.split('|')
        if len(parts) >= 2:
            last_part = parts[-1].strip()
            # Check if the last part is one of the unique collections from the spreadsheet
            unique_collections = [
                "Weatherstripping", "Exterior Rubber", "Interior Rubber and Carpets", "Interior Rubber",
                "Literature", "Uncategorized", "Front Axle", "Wheels", "Rear Axle",
                "Service Brakes", "Clutch", "Cooling", "Electrical", "Engine", "Fuel",
                "Universal Joint", "Springs", "Steering", "Transmission", "Body & Glass",
                "Accessories"
                # Add other collections from your spreadsheet here
            ]
            
            if last_part in unique_collections:
                new_title = '|'.join(parts[:-1]).strip()
                if new_title != title:
                    applied_transforms.append("Removed category section")
                    title = new_title
                    if calculate_pixel_width(title) <= max_pixels:
                        return title, applied_transforms
    
    # Step 7: Replace multiple Mopar makes with "Mopar Cars" if applicable
    mopar_makes = ["Chrysler", "DeSoto", "Dodge", "Plymouth"]
    found_makes = [make for make in mopar_makes if make in title]
    
    if len(found_makes) >= 2:
        # Create a pattern that matches consecutive Mopar makes with various separators
        make_pattern = r'\b(' + '|'.join(re.escape(make) for make in mopar_makes) + r')\b'
        
        # First identify all positions where makes appear
        make_positions = []
        for match in re.finditer(make_pattern, title):
            make_positions.append((match.start(), match.end(), match.group(1)))
        
        # Process only if we found at least 2 makes
        if len(make_positions) >= 2:
            # Sort positions by start index
            make_positions.sort()
            
            # Find consecutive makes (with separators between them)
            i = 0
            while i < len(make_positions) - 1:
                # Get current and next make positions
                current_start, current_end, current_make = make_positions[i]
                next_start, next_end, next_make = make_positions[i+1]
                
                # Check if they're close (within 15 chars - allowing for separators like ", and ")
                if next_start - current_end < 15:
                    # Extract the text between them to check if it's a valid separator
                    separator = title[current_end:next_start]
                    if re.search(r'^[\s,&]+$|^\s+and\s+$', separator):
                        # Find where this cluster of makes ends (may be more than 2)
                        cluster_end = i + 1
                        while cluster_end < len(make_positions) - 1:
                            end_pos, next_pos = make_positions[cluster_end], make_positions[cluster_end+1]
                            gap = next_pos[0] - end_pos[1]
                            if gap < 15 and re.search(r'^[\s,&]+$|^\s+and\s+$', title[end_pos[1]:next_pos[0]]):
                                cluster_end += 1
                            else:
                                break
                        
                        # Replace this cluster with "Mopar Cars"
                        cluster_start_pos = make_positions[i][0]
                        cluster_end_pos = make_positions[cluster_end][1]
                        
                        # Create new title with replacement
                        new_title = title[:cluster_start_pos] + "Mopar Cars" + title[cluster_end_pos:]
                        if new_title != title:
                            applied_transforms.append('Replaced multiple makes with "Mopar Cars"')
                            if calculate_pixel_width(new_title) <= max_pixels:
                                return new_title, applied_transforms
                            title = new_title  # Continue with other shortenings if still too long
                        
                        # Need to update our position list after replacement
                        break
                i += 1
    
    # Step 8: Replace multiple Mopar years and makes with "Classic Mopar Cars"
    year_make_pattern = r'(\d{4}-\d{4}|\d{4}).*?Mopar Cars'
    if re.search(year_make_pattern, title):
        new_title = re.sub(year_make_pattern, "Classic Mopar Cars", title)
        if new_title != title:
            applied_transforms.append('Replaced years and makes with "Classic Mopar Cars"')
            title = new_title
            if calculate_pixel_width(title) <= max_pixels:
                return title, applied_transforms
    
    return title, applied_transforms

def format_meta_title(title):
    """
    Format a meta title according to specific rules and shorten if necessary.
    """
    max_pixels = 580  # Approximate Google desktop title width limit
    transformations = []
    
    if calculate_pixel_width(title) <= max_pixels:
        return title, transformations  # Return both values even when no changes needed
    
    # Step 7: Shorten if necessary
    final_title, shortening_transforms = shorten_title(title, max_pixels)
    transformations.extend(shortening_transforms)
    
    # if not transformations:
    #     transformations.append("Applied custom formatting")
    
    return final_title, transformations

def explain_changes(original, formatted):
    """Explain what operations were performed to format the title."""
    explanations = []
    
    # Check for spaces around dashes in years
    if re.search(r'\d{4}\s+-\s+\d{4}', original) and not re.search(r'\d{4}\s+-\s+\d{4}', formatted):
        explanations.append("Removed spaces around dash in year range")
    
    # Check for "and" to "&" conversion
    if " and " in original and " & " in formatted and " and " not in formatted:
        explanations.append('Replaced "and" with "&"')
    
    # Check for comma removal
    if ',' in original and ',' not in formatted:
        explanations.append("Removed commas")
    
    # Check for Mopar makes replacement
    mopar_makes = ["Chrysler", "DeSoto", "Dodge", "Plymouth"]
    mopar_count_original = sum(1 for make in mopar_makes if make in original)
    if mopar_count_original >= 2 and "Mopar Cars" in formatted:
        explanations.append(f"Replaced multiple makes with \"Mopar Cars\"")
    
    # Check for category removal
    if "|" in original and "|" not in formatted:
        explanations.append("Removed category section")
    elif original.count("|") > formatted.count("|"):
        explanations.append("Simplified title structure")
    
    if not explanations:
        explanations.append("Applied custom formatting")
        
    return explanations

# Example usage
if __name__ == "__main__":
    titles = [
        "Dimmer Switch Grommet for 1930 - 1976 Chrysler, DeSoto, Dodge & Plymouth | Interior Rubber",
        "Custom Carpet Set for 1935-1936 Plymouth, Dodge, Chrysler Sweet like cinnamon",
        # "Tail Light Pads for 1934 - 1935 Plymouth | Exterior Rubber",
        # "Dimmer Switch Grommet for 1930 - 1976 Mopar Cars & Trucks | Interior Rubber and Carpets",
        # "Current Catalog for Classic Mopar Cars | Literature",
        # "Refundable Core Charge | Uncategorized",
        # "Control Arms Master Rebuild Kit for 1941 - 1954 Chrysler, DeSoto, Dodge & Plymouth | Front Axle",
        # "Front Outer Wheel Bearing and Cup for 1928 - 1956 Mopar | Wheels",
        # "Differential Carrier Gasket for 1928 - 1956 Chrysler, DeSoto, Dodge & Plymouth | Rear Axle",
        # "Brake Shoes & Lining Assemblies for 1933 - 1934 Dodge & Plymouth | Service Brakes",
        # "Clutch Disc - 9 1/4\" for 1935 - 1956 Plymouth | Clutch",
        # "Fan Belt for 1934 - 1953 Chrysler, DeSoto, Dodge & Plymouth | Cooling",
        # "Tune Up Kit for 1928 - 1932 Plymouth | Electrical",
        # "Intake Valve for 1933 - 1941 Plymouth & Dodge | Engine",
        # "Carburetor Rebuild Kit for 1951 - 1972 Chrysler, DeSoto, Dodge & Plymouth | Fuel",
        # "Universal Joint Boot & Clamp Kit for 1933 - 1956 Dodge & Plymouth | Universal Joint",
        # "Front Coil Spring Rubber Silencer - Upper for 1935 - 1956 Chrysler & DeSoto | Springs",
        # "Steering Box Kit for 1933 - 1934 Dodge & Plymouth | Steering",
        # "Vacamatic Transmission Diaphragm for 1941-1942 Chrysler & DeSoto | Transmission",
        # "Interior Door Handle - Left for 1939 - 1940 Plymouth | Body & Glass",
        # "Left Spotlight - Mopar Logo - Original Unity Brand 1928-1955 | Accessories"
    ]

    print("\n# Meta Title Formatting Results\n")

    for i, title in enumerate(titles, 1):
        formatted, transformations = format_meta_title(title)
        pixel_width_original = calculate_pixel_width(title)
        pixel_width_formatted = calculate_pixel_width(formatted)

        print(f"## Example #{i}")
        print()
        print(f"**ORIGINAL** ({len(title)} chars, {pixel_width_original} pixels)")
        print(f"```\n{title}\n```")
        print()
        print(f"**FORMATTED** ({len(formatted)} chars, {pixel_width_formatted} pixels)")
        print(f"```\n{formatted}\n```")

        # Calculate actual differences
        char_diff = len(title) - len(formatted)
        pixel_diff = pixel_width_original - pixel_width_formatted

        if char_diff > 0 or pixel_diff > 0:
            print()
            print(f"**REDUCED BY:** {char_diff} characters, {pixel_diff} pixels")

            # Show operations performed
            print("\n**OPERATIONS PERFORMED:**")
            for transform in transformations:
                print(f"* {transform}")

        print("\n---\n")
