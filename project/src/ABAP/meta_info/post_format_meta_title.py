import re


def normalize_brands(title):
    """Normalize brand names to proper case."""
    return re.sub(r"\bMOPAR\b", "Mopar", title)


def extract_title_parts(title):
    """Split title into main content and category if present."""
    if "|" in title:
        main_part, category = title.split("|", 1)
        return main_part.strip(), category.strip()
    return title.strip(), ""


def extract_year_range(main_part):
    """Extract and format year range from the title."""
    year_pattern = r"(?:for\s+)?(\d{4}(?:\s*-\s*\d{4})?)"
    year_match = re.search(year_pattern, main_part)

    if not year_match:
        return ""

    year_range = year_match.group(1)
    # Ensure proper spacing around dash
    if "-" in year_range:
        start_year, end_year = re.split(r"\s*-\s*", year_range)
        return f"{start_year} - {end_year}"
    return year_range


def extract_makes(main_part):
    """Extract vehicle makes from the title."""
    make_patterns = [
        r"(?:for\s+)?\d{4}(?:\s*-\s*\d{4})?\s+(.*?)(?:\||$)",
        r"for\s+(.*?)(?:\||$)",
    ]

    for pattern in make_patterns:
        make_match = re.search(pattern, main_part)
        if make_match:
            makes_text = make_match.group(1).strip()
            # Replace "Mopar Cars" with full list
            if "Mopar Cars" in makes_text:
                makes_text = makes_text.replace(
                    "Mopar Cars", "Chrysler, DeSoto, Dodge, and Plymouth"
                )

            # Extract individual makes
            return [
                make.strip()
                for make in re.split(r",|&|\band\b", makes_text)
                if make.strip()
            ]

    return []


def extract_product_name(main_part):
    """Extract product name from the beginning of the title."""
    product_match = re.match(r"^(.*?)(?:for|\d{4}|\||$)", main_part)
    if product_match:
        return product_match.group(1).strip()
    return ""


def determine_vintage_type(year_range):
    """Determine if vehicles are vintage, classic, or both based on year range."""
    if not year_range:
        return ""

    years = re.findall(r"\d{4}", year_range)
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


def shorten_title(title, max_length=60):
    """Apply progressive shortening strategies to fit the title within max_length."""
    if len(title) <= max_length:
        return title

    # Step 1: Remove spaces between dashes in years
    title = re.sub(r"(\d{4})\s+-\s+(\d{4})", r"\1-\2", title)
    if len(title) <= max_length:
        return title

    # Step 2: Replace "and" with "&"
    title = title.replace(" and ", " & ")
    if len(title) <= max_length:
        return title

    # Step 3: Remove the last comma in a list before "&"
    title = re.sub(r",\s+&", " &", title)
    if len(title) <= max_length:
        return title

    # Step 3.5: remove all the commas
    title = re.sub(r",", "", title)
    if len(title) <= max_length:
        return title

    # Step 3.6: remove the category
    if "|" in title:
        title = title.rsplit("|", 1)[0].strip()
    if len(title) <= max_length:
        return title

    # Step 4: Replace multiple Mopar makes with "Mopar Cars" if applicable
    mopar_makes = ["Chrysler", "DeSoto", "Dodge", "Plymouth"]
    found_makes = [make for make in mopar_makes if make in title]

    if len(found_makes) >= 2:
        # Create a pattern that matches consecutive Mopar makes with various separators
        # This looks for patterns like "Chrysler, Dodge", "Dodge & Plymouth", etc.
        make_pattern = (
            r"\b(" + "|".join(re.escape(make) for make in mopar_makes) + r")\b"
        )

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
                next_start, next_end, next_make = make_positions[i + 1]

                # Check if they're close (within 15 chars - allowing for separators like ", and ")
                if next_start - current_end < 15:
                    # Extract the text between them to check if it's a valid separator
                    separator = title[current_end:next_start]
                    if re.search(r"^[\s,&]+$|^\s+and\s+$", separator):
                        # Find where this cluster of makes ends (may be more than 2)
                        cluster_end = i + 1
                        while cluster_end < len(make_positions) - 1:
                            end_pos, next_pos = (
                                make_positions[cluster_end],
                                make_positions[cluster_end + 1],
                            )
                            gap = next_pos[0] - end_pos[1]
                            if gap < 15 and re.search(
                                r"^[\s,&]+$|^\s+and\s+$",
                                title[end_pos[1] : next_pos[0]],
                            ):
                                cluster_end += 1
                            else:
                                break

                        # Replace this cluster with "Mopar Cars"
                        cluster_start_pos = make_positions[i][0]
                        cluster_end_pos = make_positions[cluster_end][1]

                        # Create new title with replacement
                        new_title = (
                            title[:cluster_start_pos]
                            + "Mopar Cars"
                            + title[cluster_end_pos:]
                        )
                        if len(new_title) <= max_length:
                            return new_title
                        title = new_title  # Continue with other shortenings if still too long

                        # Need to update our position list after replacement
                        break
                i += 1

    return title


def format_meta_title(title):
    """
    Format a meta title according to specific rules and shorten if necessary.
    """
    if len(title) <= 60:
        return title
    # Step 1: Normalize brand names
    title = normalize_brands(title)

    # Step 2: Extract main components
    main_part, category = extract_title_parts(title)

    # Step 3: Extract individual components
    product_name = extract_product_name(main_part)
    year_range = extract_year_range(main_part)
    makes = extract_makes(main_part)

    # Step 4: Determine vintage/classic type (currently unused in final output)
    vintage_type = determine_vintage_type(year_range)

    # Step 5: Format makes into a string
    makes_str = format_makes_list(makes)

    # Step 6: Build the formatted title
    formatted_title = f"{product_name} for {year_range} {makes_str}"
    if len(formatted_title) > 60:
        formatted_title = f"{product_name} | {year_range} {makes_str}"
    if category:
        formatted_title += f" | {category}"

    print(f"LONGBOI_Formatted ({len(formatted_title)} chars): {formatted_title}")
    # Step 7: Shorten if necessary

    final_title = shorten_title(formatted_title)

    return final_title


# Example usage
if __name__ == "__main__":
    titles = [
        "Convertible Top Header Rubber Seal for 1960 - 1962 Chrysler, DeSoto & Plymouth | Weatherstripping",
        "Tail Light Pads for 1934 - 1935 Plymouth | Exterior Rubber",
        "Dimmer Switch Grommet for 1930 - 1976 Mopar Cars & Trucks | Interior Rubber and Carpets",
        "Current Catalog for Classic Mopar Cars | Literature",
        "Refundable Core Charge | Uncategorized",
        "Control Arms Master Rebuild Kit for 1941 - 1954 Chrysler, DeSoto, Dodge & Plymouth | Front Axle",
        "Front Outer Wheel Bearing and Cup for 1928 - 1956 Mopar | Wheels",
        "Differential Carrier Gasket for 1928 - 1956 Chrysler, DeSoto, Dodge & Plymouth | Rear Axle",
        "Brake Shoes & Lining Assemblies for 1933 - 1934 Dodge & Plymouth | Service Brakes",
        'Clutch Disc - 9 1/4" for 1935 - 1956 Plymouth | Clutch',
        "Fan Belt for 1934 - 1953 Chrysler, DeSoto, Dodge & Plymouth | Cooling",
        "Tune Up Kit for 1928 - 1932 Plymouth | Electrical",
        "Intake Valve for 1933 - 1941 Plymouth & Dodge | Engine",
        "Carburetor Rebuild Kit for 1951 - 1972 Chrysler, DeSoto, Dodge & Plymouth | Fuel",
        "Universal Joint Boot & Clamp Kit for 1933 - 1956 Dodge & Plymouth | Universal Joint",
        "Front Coil Spring Rubber Silencer - Upper for 1935 - 1956 Chrysler & DeSoto | Springs",
        "Steering Box Kit for 1933 - 1934 Dodge & Plymouth | Steering",
        "Vacamatic Transmission Diaphragm for 1941-1942 Chrysler & DeSoto | Transmission",
        "Interior Door Handle - Left for 1939 - 1940 Plymouth | Body & Glass",
        "Left Spotlight - Mopar Logo - Original Unity Brand 1928-1955 | Accessories",
    ]

    for title in titles:
        formatted = format_meta_title(title)
        print(f"Original ({len(title)} chars): {title}")
        print(f"Formatted ({len(formatted)} chars): {formatted}")
        print("--------------------------------")
