from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int
    category: str = "general"


@dataclass
class LineItem:
    product: Product
    qty: int

    def __post_init__(self):
        if self.qty <= 0:
            raise ValueError(f"quantity must be positive, got {self.qty}")


@dataclass
class Order:
    customer: str
    items: list[LineItem] = field(default_factory=list)

    def add(self, product: Product, qty: int = 1) -> None:
        for item in self.items:
            if item.product.sku == product.sku:
                item.qty += qty
                return
        self.items.append(LineItem(product, qty))

    @property
    def n_units(self) -> int:
        return sum(item.qty for item in self.items)
