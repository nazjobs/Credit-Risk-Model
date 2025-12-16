from pydantic import BaseModel


class TransactionInput(BaseModel):
    ProviderId: int
    ProductId: int
    ProductCategory: int
    ChannelId: int
    Amount: float
    Value: int
    PricingStrategy: int
    FraudResult: int
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
