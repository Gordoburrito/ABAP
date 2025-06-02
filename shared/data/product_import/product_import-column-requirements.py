REQUIRED_ALWAYS = ["Title", "Body HTML", "Vendor", "Tags", "Image Src", "Image Command", "Image Position", "Image Alt Text", "Variant Barcode", "Variant Price", "Variant Cost", "Metafield: title_tag [string]", "Metafield: description_tag [string]"]
REQUIRED_MULTI_VARIANTS = ["Option1 Name", "Option1 Value", "Option2 Name", "Option2 Value", "Option3 Name", "Option3 Value", "Variant Position", "Variant Image"]
REQUIRED_MANUAL = ["Command", "Tags Command", "Custom Collections", "Variant Command", "Variant SKU", "Variant HS Code", "Variant Country of Origin","Variant Metafield: harmonized_system_code [string]"]
NO_REQUIREMENTS = ["ID", "Category: ID", "Category: Name", "Category", "Smart Collections", "Image Type", "Image Width", "Image Height", "Variant Inventory Item ID", "Variant ID", "Variant Weight", "Variant Weight Unit", "Variant Compare At Price","Variant Taxable", "Variant Tax Code", "Variant Inventory Tracker", "Variant Inventory Policy", "Variant Fulfillment Service", "Variant Requires Shipping", "Variant Inventory Qty", "Variant Inventory Adjust", "Variant Province of Origin", "Metafield: custom.engine_types [list.single_line_text_field]", "Metafield: mm-google-shopping.custom_product [boolean]", "Variant Metafield: mm-google-shopping.custom_label_4 [single_line_text_field]", "Variant Metafield: mm-google-shopping.custom_label_3 [single_line_text_field]", "Variant Metafield: mm-google-shopping.custom_label_2 [single_line_text_field]", "Variant Metafield: mm-google-shopping.custom_label_1 [single_line_text_field]", "Variant Metafield: mm-google-shopping.custom_label_0 [single_line_text_field]", "Variant Metafield: mm-google-shopping.size_system [single_line_text_field]", "Variant Metafield: mm-google-shopping.size_type [single_line_text_field]", "Variant Metafield: mm-google-shopping.mpn [single_line_text_field]", "Variant Metafield: mm-google-shopping.gender [single_line_text_field]", "Variant Metafield: mm-google-shopping.condition [single_line_text_field]", "Variant Metafield: mm-google-shopping.age_group [single_line_text_field]", "Metafield: mm-google-shopping.mpn [single_line_text_field]"]

cols = "ID,Command,Title,Body HTML,Vendor,Tags,Tags Command,Category: ID,Category: Name,Category,Custom Collections,Smart Collections,Image Type,Image Src,Image Command,Image Position,Image Width,Image Height,Image Alt Text,Variant Inventory Item ID,Variant ID,Variant Command,Option1 Name,Option1 Value,Option2 Name,Option2 Value,Option3 Name,Option3 Value,Variant Position,Variant SKU,Variant Barcode,Variant Image,Variant Weight,Variant Weight Unit,Variant Price,Variant Compare At Price,Variant Taxable,Variant Tax Code,Variant Inventory Tracker,Variant Inventory Policy,Variant Fulfillment Service,Variant Requires Shipping,Variant Inventory Qty,Variant Inventory Adjust,Variant Cost,Variant HS Code,Variant Country of Origin,Variant Province of Origin,Metafield: title_tag [string],Metafield: description_tag [string],Metafield: custom.engine_types [list.single_line_text_field],Metafield: mm-google-shopping.custom_product [boolean],Variant Metafield: mm-google-shopping.custom_label_4 [single_line_text_field],Variant Metafield: mm-google-shopping.custom_label_3 [single_line_text_field],Variant Metafield: mm-google-shopping.custom_label_2 [single_line_text_field],Variant Metafield: mm-google-shopping.custom_label_1 [single_line_text_field],Variant Metafield: mm-google-shopping.custom_label_0 [single_line_text_field],Variant Metafield: mm-google-shopping.size_system [single_line_text_field],Variant Metafield: mm-google-shopping.size_type [single_line_text_field],Variant Metafield: mm-google-shopping.mpn [single_line_text_field],Variant Metafield: mm-google-shopping.gender [single_line_text_field],Variant Metafield: mm-google-shopping.condition [single_line_text_field],Variant Metafield: mm-google-shopping.age_group [single_line_text_field],Variant Metafield: harmonized_system_code [string],Metafield: mm-google-shopping.mpn [single_line_text_field]"

cols_list = cols.split(",")
category_cols = NO_REQUIREMENTS + REQUIRED_ALWAYS + REQUIRED_MULTI_VARIANTS + REQUIRED_MANUAL

# Check if all columns in cols_list are in category_cols
missing_cols = set(cols_list) - set(category_cols)
if missing_cols:
    print(f"Warning: The following columns are in cols_list but not in category_cols: {missing_cols}")

# Check if all columns in category_cols are in cols_list
extra_cols = set(category_cols) - set(cols_list)
if extra_cols:
    print(f"Warning: The following columns are in category_cols but not in cols_list: {extra_cols}")
else:
    print("All columns in category_cols are in cols_list")






