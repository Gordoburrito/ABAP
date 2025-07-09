from pydantic import BaseModel, Field

class ProductData(BaseModel):
    title: str = Field(description="Title of the part/ product")
    year_min: int = Field(description="Minimum year of vehicle compatibility")
    year_max: int = Field(description="Maximum year of vehicle compatibility")
    make: str = Field(description="Make of the vehicle compatibility")
    model: list[str] = Field(description="List of vehicle models compatible with this product")
    mpn: str = Field(description="Manufacturer Part Number (SKU)")
    cost: float = Field(description="Vendor cost amount")
    price: float = Field(description="Suggested retail price")
    body_html: str = Field(description="Long description of the product including features, compatibility, and benefits") 
    collection: str = Field(description="Collection of the product")
    product_type: str = Field(description="Product type of the product")
    meta_title: str = Field(description="Meta title of the product")
    meta_description: str = Field(description="Meta description of the product")