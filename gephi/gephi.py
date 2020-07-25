#/usr/local/bin/jython

# adapted from https://github.com/jsundram/pygephi

import java
import os, glob, sys
import classpath
path = os.path.realpath(__file__)
#toolkit = os.path.join(os.path.dirname(path), 'gephi-toolkit-0.8.7.jar')
toolkit = os.path.join(os.path.dirname(path), 'gephi-toolkit-0.9.1-all.jar')
classpath.addFile(toolkit)
# TODO: need /Users/jason/Code/gephi/toolkit/gephi-toolkit/org-openide-util.jar?

# necessary for save - there should be a less scattershot way to do this?
#for j in glob.glob('/Users/jason/Code/gephi/platform/platform/modules/*.jar'): classpath.addFile(j)

import org.openide.util.Lookup as Lookup
Lookup = Lookup.getDefault().lookup

STEPS = 50000


def lookup(name, namespace='org.gephi.'):
    return Lookup(java.lang.Class.forName(namespace + name))

import org.gephi.project.api.ProjectController as ProjectController

ProjectController = lookup('project.api.ProjectController')
ExportController = lookup('io.exporter.api.ExportController')
ImportController = lookup('io.importer.api.ImportController')
GraphController = lookup('graph.api.GraphController')
PreviewController = lookup('preview.api.PreviewController')


"""
http://pastebin.com/nFX0jHFV
import org.gephi.project.api as project
import org.gephi.data.attributes.api as attributes
import org.gephi.filters.api as filters
import org.gephi.graph.api as graph
import org.gephi.io.exporter.api as exporter
import org.gephi.io.generator.api as generator
import org.gephi.io.importer.api as importer
import org.gephi.layout.api as layout
import org.gephi.partition.api as partition
import org.gephi.preview as preview
import org.gephi.project.api as project
import org.gephi.ranking.api as ranking
import org.gephi.statistics as statistics
import org.gephi.utils as utils
"""
def show():
    """ adapted from here: http://wiki.gephi.org/index.php/Toolkit_-_Reuse_the_Preview_Applet"""
    from javax.swing import JFrame
    from java.awt import BorderLayout

    pc = PreviewController
    pc.refreshPreview()
     
    # New Processing target, get the PApplet
    target = pc.getRenderTarget("processing")
    applet = target.getApplet()
    applet.init()
     
    # Refresh the preview and reset the zoom
    try:
        pc.render(target) 
    except Exception:
        # throws sun.dc.pr.PRError: sun.dc.pr.PRError: setPenT4: invalid pen transformation (singular)
        pass
    target.refresh()
    target.resetZoom()
     
    # Add the applet to a JFrame and display
    frame = JFrame("Preview")
    frame.setLayout(BorderLayout())
     
    # frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.add(applet, BorderLayout.CENTER)
     
    frame.pack()
    frame.setVisible(True)

def adjust_sizes(nodes, attribute="in_degree", scale=None, in_degrees=None):
    node_attributes = nodes[0].getNodeData().attributes.values

    if not scale:
        #m = max([n.getNodeData().attributes.values[ix].value for n in nodes])
        m = in_degrees[max(in_degrees)]
        scale = lambda f : m-f

    for node in nodes:
        node_data = node.getNodeData()
        value = in_degrees[str(node)]
        #value = float(node_data.attributes.values[ix].value)
        node_data.setSize(1 + scale(value)) # avoid problem with 0 sizes
        # print "changed %s size from %s to %s" % (node_data.getLabel(), value, (1 + scale(value)))

#def layout(graph_file='./90-03-24.dot', out_gexf='./new_graph.gexf', layout=True, save_pdf=True, save_gephi=False, in_degrees=None):
def layout(graph_file=None, layout=True, save_pdf=True, save_gephi=False, in_degrees=None):
    print "loading graph"
    # tips:
    #   http://wiki.gephi.org/index.php/Toolkit_portal
    #   it helps to open the jar to see what's in it.
    #   work through stuff here: https://wiki.gephi.org/index.php/Toolkit_-_Headless_Gephi
    #   look at gephi-toolkit demos and look for gexf import
    #   find out where shit is by cloning git://github.com/gephi/gephi.git and grepping
    #   Layout http://gephi.org/docs/toolkit/org/gephi/layout/plugin/forceAtlas/ForceAtlasLayout.html
    #   Plugins: https://wiki.gephi.org/index.php/How_to_use_Gephi_plug-ins_with_the_Gephi_Toolkit
    ProjectController.newProject() 
    w = ProjectController.getCurrentWorkspace()

    graph = ImportController.importFile(java.io.File(graph_file))

    edge_default = graph.getEdgeDefault()
    graph.setEdgeDefault(edge_default.values()[0])#DIRECTED, UNDIRECTED, MIXED

    dp = lookup('io.processor.plugin.DefaultProcessor')
    ImportController.process(graph, dp, w)

    if layout:
        # Setup Layout
        fa = lookup('layout.plugin.forceAtlas.ForceAtlas')
        force_atlas = fa.buildLayout()
        force_atlas.setGraphModel(GraphController.getGraphModel())

        force_atlas.resetPropertiesValues() # set defaults
        # for p in sorted(a, key=lambda p: (p.category, p.displayName, p.name)): 
        #    print "%s -> '%s' (%s) = %s" % (p.category, p.displayName, p.name, p.value)
        #force_atlas.adjustSizes = True
        #force_atlas.attractionStrength = 8
        #force_atlas.repulsionStrength = 2000
        #print(dir(force_atlas))

        force_atlas.inertia = 0.1
        force_atlas.repulsionStrength = 230
        force_atlas.attractionStrength = 10
        force_atlas.maxDisplacement = 10
        force_atlas.gravity = 20
        force_atlas.speed = 1
        force_atlas.adjustSizes = True
        

        force_atlas.initAlgo()
        steps = STEPS
        print "running force atlas for %d steps" % steps
        while steps and force_atlas.canAlgo() and not force_atlas.isConverged():
            if steps % 5000 == 0:
                print "step %d" % steps
            force_atlas.goAlgo()
            steps -= 1


        
        # I think there's a bug in label adjust so that it doesn't work in the toolkit?
        # http://forum.gephi.org/viewtopic.php?t=1435
        lab = lookup('layout.plugin.labelAdjust.LabelAdjustBuilder')
        label_adjust = lab.buildLayout()
        label_adjust.resetPropertiesValues()
        label_adjust.setGraphModel(GraphController.getGraphModel())

        
        label_adjust.initAlgo()
        steps = 50000
        print "running label adjust for %d steps" % steps
        while steps and label_adjust.canAlgo() and not label_adjust.isConverged():
            if steps % 5000 == 0:
                print "step %d" % steps
            label_adjust.goAlgo()
            steps -= 1
        

    # Adjust node sizes (Force Atlas changes them?)
    gm = GraphController.getGraphModel()
    gv = gm.getGraphVisible()
    nodes = list(gv.getNodes())
    #adjust_sizes(nodes, in_degrees=in_degrees)

    # adjust the look a bit for the pdf
    if not in_degrees:
        for node in nodes:
            #node_data = node.getNodeData()
            #node_data.setSize(10)
            node.setSize(8)
    

    preview = PreviewController.getModel()
    preview.properties.putValue('node.border.width', 0)
    preview.properties.putValue('node.label.show', True)
    preview.properties.putValue('node.label.font', java.awt.Font('Arial', java.awt.Font.PLAIN, 3))
    preview.properties.putValue('edge.curved', False)
    #preview.properties.putValue('directed', True)
    preview.properties.putValue('arrow.size', 3)


    PreviewController.refreshPreview() # necessary?
    
    # When done, save what we did
    # 1) GEXF
    #ExportController.exportFile(java.io.File(out_gexf))
    
    # 2) PDF
    if save_pdf:
        # reset node sizes, small ones cause problems for pdf render
        #adjust_sizes(nodes, scale=lambda f : 4 * (4-f), in_degrees=in_degrees)
        #part, ext = os.path.splitext(out_gexf)
        part, ext = os.path.splitext(graph_file)
        out_pdf = part + '_gephi.pdf'
        pdf = ExportController.getExporter("pdf")
        pdf.landscape = True
        ExportController.exportFile(java.io.File(out_pdf), pdf)

    # 3) GEPHI (how do you save a gephi file?)
    #if save_gephi:
    #    saver = ProjectController.saveProject(ProjectController.getCurrentProject(), java.io.File("new_graph.gephi"))
    #    saver.run()
    

def inDegrees(dot):
    # TODO: replace with gephi function get n degrees
    with open(dot) as f:
        lines = f.readlines()
    in_degrees = {}
    for l in lines:
        s = l.split('"')
        if len(s) < 4:
            continue
        n1 = s[1]
        n2 = s[3]
        for n in [n1, n2]:
            if n not in in_degrees:
                in_degrees[n] = 0
        if '->' in l:
            in_degrees[n2] += 1
    return in_degrees

    

if __name__ == '__main__':
    # from optparse import OptionParser
    # parser = OptionParser()
    try:
        #layout(sys.argv[1], sys.argv[2])
        #layout(sys.argv[1], in_degrees=inDegrees(sys.argv[1]))
        layout(sys.argv[1])
    except IndexError:
        #layout(in_degrees=inDegrees())
        print('NO INPUT FILE SPECIFIED')
        
    
