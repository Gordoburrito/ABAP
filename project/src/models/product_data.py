from pydantic import BaseModel, Field

class ProductData(BaseModel):
    title: str = Field(description="Title of the part/ product")
    year_min: int = Field(description="Minimum year of vehicle compatibility")
    year_max: int = Field(description="Maximum year of vehicle compatibility")
    make: str = Field(description="Make of the vehicle compatibility")
    model: str = Field(description="Model of the vehicle compatibility")
    mpn: str = Field(description="Manufacturer Part Number (SKU)")
    cost: float = Field(description="Vendor cost amount")
    price: float = Field(description="Suggested retail price")
    body_html: str = Field(description="Long description of the product including features, compatibility, and benefits") 