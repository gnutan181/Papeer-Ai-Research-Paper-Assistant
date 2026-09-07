from typing import Literal

from pydantic import BaseModel


class FigureAnalysis(BaseModel):
    asset_type: Literal[
        "chart", "diagram", "photograph", "table_image", "equation", "other"
    ]
    summary: str
    visible_text: list[str] = []
    axes: list[str] = []
    legend_items: list[str] = []
    visible_values: list[str] = []
    trends: list[str] = []
    relationships: list[str] = []
    conclusion: str = ""
    uncertainties: list[str] = []
