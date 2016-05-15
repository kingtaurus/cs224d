import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

confusion = np.array([[42452,    27,    45,   175,    60],
	           [  255,  1636,    12,   152,    39],
	           [  317,    26,   863,    42,    20],
	           [  598,    73,    31,  1319,    71],
	           [  546,    24,     3,    49,  2527]], dtype=np.int32)

cm = confusion.copy()
cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
# cm *= 255
# cm = cm.astype('uint8')
cm = cm[:, :]
print(cm)

plt.figure()
cmap1 = mpl.colors.ListedColormap(sns.color_palette("coolwarm", 100))
# print(sns.color_palette(sns.color_palette("coolwarm", 100)))

#plt.cm.Blues
plt.imshow(cm, interpolation='nearest', cmap=cmap1)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(["a", "b", "c", "d", "e"]))
plt.xticks(tick_marks, ["a", "b", "c", "d", "e"], rotation=45)
plt.yticks(tick_marks, ["a", "b", "c", "d", "e"])
plt.gca().xaxis.grid(b=False)
plt.gca().yaxis.grid(b=False)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cm.png')
plt.show()




# conf_matrix = tf.image_summary("confusion_matrix" + str(epoch), tf.convert_to_tensor(confusion.astype(np.float32)))
# conf_summary = session.run(conf_matrix)
# model.summary_writer.add_summary(conf_summary, epoch)
