from __init__ import *
import torch.nn.modules.loss as loss

class TeacherStudentCELoss(loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, student_output, teacher_output):
        teacher_logits = teacher_output.logits
        student_logits = student_output.logits
        