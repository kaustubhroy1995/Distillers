from __init__ import *
import torch.nn.modules.loss as loss

class TeacherStudentCELoss(loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.ce_loss = nn.KLDivLoss()
    def forward(self, student_output, teacher_output):
        """
        Huggingface models output as an Output Object that contains "logits", "attentions" and "hidden_states" attributes
        for crossentropy, we wish to compare the student and teacher logits in the pooler output.
        Other methods of distillation use other outputs of the teacher and student models to compare when calculating
        distillation losses
        """
        teacher_logits = teacher_output.logits
        student_logits = student_output.logits

class TeacherStudentCosineEmbeddingLoss(loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', layer_map=None) -> None:
        """
        Here we encounter a new attribute called the layer_map
        Generally speaking, teacher model and student model will have different number of layers
        So, in order to compare outputs of the embedding vectors of hidden layers from student and teacher outputs
        we must first design a way to find the mapping of which student layer to map to which teacher layer.
        It may not be a one-to-one map. you can map any linear combination of student layers to any 
        linear combination of teacher layers. the layer_map must therefore be a NxM matrix;
        where N is the number of student layers and M is the number of teacher layers.

        e.g:

        suppose the student model has 4 layers and the teacher model has 8 layers, and we want every student layer to
        map to the arithmetic mean of every two teacher layers' hidden output:
        layer_map = |   0.5     0.5     0       0       0       0       0       0       |
                    |   0       0       0.5     0.5     0       0       0       0       |
                    |   0       0       0       0       0.5     0.5     0       0       |
                    |   0       0       0       0       0       0       0.5     0.5     |
        """
        super().__init__(size_average, reduce, reduction)
        self.layer_map = layer_map
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, student_output, teacher_output):
        """
        """
        student_hidden_layers = student_output.hidden_states
        teacher_hidden_layers = teacher_output.hidden_states
        for idx, weight in np.ndenumerate(self.layer_map):
            loss += weight*self.loss_fn(student_hidden_layers[idx[0]], teacher_hidden_layers[idx[1]])
        return loss
        