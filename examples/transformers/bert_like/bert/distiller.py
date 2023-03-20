from __init__ import *


class Distiller(nn.Module):
    def __init__(
        self, teacher_model, student_model, distillation_type, alpha, beta
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.beta = beta
        self.trainable_params = [p for p in student_model.parameters() if p.requires_grad==True]
        self.opt = torch.optim.AdamW(params = [p for p in student_model.parameters() if p.requires_grad==True], lr=0.001)
    def train(self, train_data, test_data=None, opt=None, lr=0.001, learning_scheduler=None, epochs=10):
        for epoch in range(epochs):
            self.train_one_epoch(train_data, test_data, self.opt)