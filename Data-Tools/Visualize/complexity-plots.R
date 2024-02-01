rm(list = ls())
cat("\014")

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)
library(scales) # to access break formatting functions


NAMES = c('10-fold cv','90/10','70/30','50/50')
SHAPE <- c(21,24,22,25)
cb_palette <- c('#D81B60','#1E88E5','#FFC107','#004D40')
TSIZE <- 22
REPLICATES <- 30
data_dir <- './'
task_id_lists <- c(167104,167184,167168,167161,167185,189905)

p_theme <- theme(
  plot.title = element_text( face = "bold", size = 22, hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  legend.title=element_text(size=22),
  legend.text=element_text(size=23),
  axis.title = element_text(size=23),
  axis.text = element_text(size=19),
  legend.position="bottom",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  size = 0.5, linetype = "solid")
)

comp <- read.csv(paste(data_dir, 'complexity.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)
comp$acro <- factor(comp$acro, levels = NAMES)

# plot for task 167104
task_1 = filter(comp, taskid == task_id_lists[1]) %>%
  ggplot(., aes(x = acro, y = complexity, color = acro, fill = acro, shape = acro)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
  scale_y_log10(
    name="Complexity",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_x_discrete(
    name="Treatment"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  p_theme

# plot for task 167184
task_2 = filter(comp, taskid == task_id_lists[2]) %>%
  ggplot(., aes(x = acro, y = complexity, color = acro, fill = acro, shape = acro)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
  scale_y_log10(
    name="Complexity",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x)),
    position = 'right',
    sec.axis = dup_axis()
  ) +
  scale_x_discrete(
    name="Treatment"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  p_theme +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x.top = element_blank(), # remove ticks/text on labels
        axis.ticks.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.title.x.bottom = element_blank(), # remove titles
        axis.title.y.left = element_blank())


# plot for task 167168
task_3 = filter(comp, taskid == task_id_lists[3]) %>%
  ggplot(., aes(x = acro, y = complexity, color = acro, fill = acro, shape = acro)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
  scale_y_log10(
    name="Complexity",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x)),
  ) +
  scale_x_discrete(
    name="Treatment"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  p_theme +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x.top = element_blank(), # remove ticks/text on labels
        axis.ticks.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.title.x.bottom = element_blank(), # remove titles
        axis.title.y.left = element_blank())

# plot for task 167161
task_4 = filter(comp, taskid == task_id_lists[4]) %>%
  ggplot(., aes(x = acro, y = complexity, color = acro, fill = acro, shape = acro)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
  scale_y_log10(
    name="Complexity",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x)),
    position = 'right',
    sec.axis = dup_axis()

  ) +
  scale_x_discrete(
    name="Treatment"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  p_theme +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x.top = element_blank(), # remove ticks/text on labels
        axis.ticks.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.title.x.bottom = element_blank(), # remove titles
        axis.title.y.left = element_blank())

# plot for task 167185
task_5 = filter(comp, taskid == task_id_lists[5]) %>%
  ggplot(., aes(x = acro, y = complexity, color = acro, fill = acro, shape = acro)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
  scale_y_log10(
    name="Complexity",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x)),

  ) +
  scale_x_discrete(
    name="Treatment"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  p_theme

# plot for task 189905
task_6 = filter(comp, taskid == task_id_lists[6]) %>%
  ggplot(., aes(x = acro, y = complexity, color = acro, fill = acro, shape = acro)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_point(position = position_jitter(width = .015, height = .0001), size = 2.0, alpha = 1.0) +
  scale_y_log10(
    name="Complexity",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x)),
    position = 'right',
    sec.axis = dup_axis()

  ) +
  scale_x_discrete(
    name="Treatment"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  p_theme +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x.top = element_blank(), # remove ticks/text on labels
        axis.ticks.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.title.x.bottom = element_blank(), # remove titles
        axis.title.y.left = element_blank())

# legend
legend <- cowplot::get_legend(
  task_1 +
    guides(
      shape=guide_legend(nrow=1,title="Evaluation strategy"),
      color=guide_legend(nrow=1,title="Evaluation strategy"),
      fill=guide_legend(nrow=1,title="Evaluation strategy")
    ) +
    theme(
      legend.position = "top",
      legend.box="verticle",
      legend.justification="center"
    )
)

row1 = plot_grid(
  task_1 + ggtitle("Task 167104") +
    theme(legend.position = "none", axis.title.x=element_blank(),axis.title.y=element_blank(),
          axis.ticks.x = element_blank(), axis.text.x = element_blank(), axis.text.y = element_text(angle = 90, hjust = 0.5)),
  task_2 + ggtitle("Task 167184") +
    theme(legend.position = "none", axis.title.x=element_blank(), axis.ticks.x = element_blank(), axis.text.x = element_blank(),
          axis.text.y = element_text(angle = 90, hjust = 0.5)),
  ncol=2,
  rel_widths = c(1.2,1.1),
  labels = c('    a','    b'),
  label_size = TSIZE
)

row2 = plot_grid(
  task_3 + ggtitle("Task 167168") +
    theme(legend.position = "none", axis.title.x=element_blank(),axis.title.y=element_blank(),
          axis.ticks.x = element_blank(), axis.text.x = element_blank(), axis.text.y = element_text(angle = 90, hjust = 0.5)),
  task_4 + ggtitle("Task 167161") +
    theme(legend.position = "none", axis.title.x=element_blank(), axis.ticks.x = element_blank(), axis.text.x = element_blank(),
          axis.text.y = element_text(angle = 90, hjust = 0.5)),
  ncol=2,
  rel_widths = c(1.2,1.1),
  labels = c('    c','    d'),
  label_size = TSIZE
)

row3 = plot_grid(
  task_5 + ggtitle("Task 167185") +
    theme(legend.position = "none", axis.title.x=element_blank(),axis.title.y=element_blank(),
          axis.ticks.x = element_blank(), axis.text.x = element_blank(), axis.text.y = element_text(angle = 90, hjust = 0.5)),
  task_6 + ggtitle("Task 189905") +
    theme(legend.position = "none", axis.title.x=element_blank(), axis.ticks.x = element_blank(), axis.text.x = element_blank(),
          axis.text.y = element_text(angle = 90, hjust = 0.5)),
  ncol=2,
  rel_widths = c(1.2,1.1),
  labels = c('    e','    f'),
  label_size = TSIZE
)

fig = plot_grid(
  ggdraw() + draw_label("Complexity of final model per OpenML task", fontface='bold', size = 24) + p_theme,
  row1,
  row2,
  row3,
  legend,
  nrow=5,
  rel_heights =  c(.1,1,1,1.1,.08),
  label_size = TSIZE
)

save_plot(
  paste(filename ="complxity.pdf"),
  fig,
  base_width=10,
  base_height=17
)
